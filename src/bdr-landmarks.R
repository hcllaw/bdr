library(docopt)

'Usage:
   bdr-landmarks.R [-l lengthscale] [--eta eta] [--scale scale] [-R Rmatrix] [-s sigma] [-k landmarks] [-d dataset] [-i iter] [-r recompile] [-T test] [-m bdr-landmarks-conjugacy] [-g global-maen] [-S standardize] [-C covariance] [--adapt adapt] [--chains chains]

Options:
   -l lengthscale [default: 1]
   -s sigma [default: 1]
   --scale scale [default: 1]
   --eta eta [default: 1]
   -k landmarks [default: 30]
   -d dataset [default: ~/bdr/data/chi2-manual-5]
   -i iter [default: 200]
   -r recompile [default: false]
   -R R matrix [default: nonstationary]
   -T include testing data? [default: false]
   -m model name [default: bdr-landmarks-conjugacy]
   -g shrink towards a global mean? [default: true]
   -S standardize labels [default: false]
   -C covariance [default: empirical]
   --chains chains [default: 4]
   --adapt adapt_delta [default: 0.8]
' -> doc

opts = docopt(doc)
ss = opts[1:(length(opts)/2)]; ss[["-d"]] = basename(ss[["-d"]]); opts.str = paste(names(ss),ss,sep=":",collapse="")
print(opts.str)
opts$scale = as.numeric(opts$scale) # this is the eta that scales R
opts$eta = as.numeric(opts$eta) # this is the SD of the Gaussian measure
opts$s = as.numeric(opts$s)
opts$k = as.numeric(opts$k)
opts$l = as.numeric(opts$l)
opts$i = as.numeric(opts$i)
opts$chains = as.integer(opts$chains)
library(data.table)
library(rstan)
library(kernlab)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

print("Reading in data...")
train = fread(sprintf("%s/train_x.csv",opts$d),header=F)
table(table(train$V6))

test = fread(sprintf("%s/test_x.csv",opts$d),header=F)
validate = fread(sprintf("%s/val_x.csv",opts$d),header=F)
train.y = read.csv(sprintf("%s/train_y.csv",opts$d),header=F)[,1]
test.y = read.csv(sprintf("%s/test_y.csv",opts$d),header=F)[,1]
validate.y = read.csv(sprintf("%s/val_y.csv",opts$d),header=F)[,1]

if(opts$S == "true") {  # turn labels into z-scores
  muy = mean(train.y)
  sdy = sd(train.y)
  train.y = (train.y - muy) / sdy
  test.y = (test.y - muy) / sdy
  validate.y = (validate.y - muy) / sdy
}

d = ncol(train)-1
print("Calculating median heuristic...")
if(nrow(train) > 1000) {
  tmp = train[sample(nrow(train[,1:d]),1000),1:d]
}  else {
  tmp = train
}
lengthscale = opts$l * median(as.numeric(as.matrix(dist(tmp))))

K = opts$k
set.seed(1)
uii = sample(nrow(train),K)
if(opts$T) { # include testing data
  X = rbind(data.matrix(train[,1:d,with=F]),data.matrix(validate[,1:d,with=F]),data.matrix(test[,1:d,with=F]))
} else {
  X = rbind(data.matrix(train[,1:d,with=F]),data.matrix(validate[,1:d,with=F])) 
}

print("Featurizing...")
u = X[uii,]
tryCatch({
  u = data.matrix(fread(sprintf("%s/landmarks-%d.csv",opts$d,opts$k),header=F))
  print("successfully read in landmark points")
}, error = function(e) {
    print(sprintf("failed to read %s/landmarks-%d.csv",opts$d,opts$k))
  print("randomly chose landmark points")
})

if(opts$R == "stationary") { # the "wrong" formulation---just double the lengthscale.
    print("Using R: stationary")
  R = opts$scale * kernelMatrix(rbfdot(.5 / (sqrt(2)*lengthscale)^2),u) 
} else if(opts$R == "nonstationary") { # the "correct" formulation, following Flaxman et al UAI 2016
    print("Using R: nonstationary")
  m = nrow(u)
  Rplus = exp(-.5 * outer(1:m,1:m,Vectorize(function(i,j) .25 * sum((u[i,]+u[j,])^2))) / (.5 * lengthscale + opts$eta^2))
  R = opts$scale * kernelMatrix(rbfdot(.5 / (sqrt(2)*lengthscale)^2),u) * Rplus
} else { # R = K
    print("Using R = K")
  R = opts$scale * kernelMatrix(rbfdot(.5 / lengthscale^2),u) 
} 
R = R + diag(1e-6,nrow(u))  # add some jitter to preserve positive definiteness
phi = kernelMatrix(rbfdot(.5 / lengthscale^2),X,u)

bag_index = c(as.integer(unlist(train[,d+1,with=F]))+1,
              as.integer(unlist(validate[,d+1,with=F])) + length(train.y) + 1)
if(opts$T)
  bag_index = c(bag_index,as.integer(unlist(test[,d+1,with=F])) + length(validate.y) + length(train.y) + 1)

mu = data.table(phi)[, lapply(.SD, mean), by=bag_index]
mu[, bag_index := NULL]
mu = data.matrix(mu)
p = length(unique(bag_index))

d = ncol(phi)
stan_data = list(d = d,
                 p = p,
                 ntrain=length(train.y), 
                 mu=mu,
                 y=train.y,
                 ytrue=c(train.y,validate.y))

if(opts$g == "true") { # use a global mean
  mu0 = colMeans(phi)
} else {
  mu0 = rep(0,opts$k)
}
if(opts$C == "empirical") {
  print("using empirical covariance")
  Sigma0 = cov(phi) 
} else {
  Sigma0 = diag(opts$s,d)
  print("using diagonal covariance")
}

if(grepl("conjugacy",opts$m) | grepl("shrinkage",opts$m)) { 
  Sigma = array(0, dim=c(p,d,d))
  
  for(i in 1:p) {
    bag = phi[bag_index == i,]
    n = nrow(bag)
    if(is.null(n)) # just one row
      n = 1

    mu[i,] = R %*% solve(R+Sigma0/n,mu[i,] - mu0) + mu0
    Sigma[i,,] = R - R %*% solve(R+Sigma0/n,R)
  }
  stan_data$mu = mu
  stan_data$Sigma = Sigma
}

if(opts$T) {
  stan_data$ntrain = length(train.y) + length(validate.y)
  stan_data$y = c(train.y,validate.y)
  stan_data$ytrue = c(train.y,validate.y,test.y)
  heldout.y = test.y 
} else {
  heldout.y = validate.y
}

mfile = sprintf("%s-compiled.rdata",opts$m)
if(opts$r == "true") { # recompile?
  print("Recompiling...")
  m = stan_model(sprintf("%s.stan",opts$m)) #bdr-landmarks-conjugacy.stan")
  save(m,file=mfile) #"bdr-landmarks-conjugacy-compiled.rdata")
} else {
  load(mfile) #"bdr-landmarks-conjugacy-compiled.rdata")
}

### cleanup
print("cleaning up...")
rm(phi,X,train,validate,test)

ptm = proc.time()
fit = sampling(m,data=stan_data,iter=opts$i,warmup=round(opts$i/2),chains=opts$chains,control=list(adapt_delta=as.numeric(opts$adapt)))

elapsed = as.numeric((proc.time() - ptm)[3])
print(fit,"beta")

yhat = colMeans(extract(fit,"yhat")$yhat)
n = stan_data$ntrain
N = stan_data$p

# check calibration
library(matrixStats)
y = c(stan_data$y,heldout.y)

yhat.ui95 = colQuantiles(extract(fit,"yhat")$yhat,probs=c(.025,.975))
within95 = yhat.ui95[,1] <= y & yhat.ui95[,2] >= y

yhat.ui80 = colQuantiles(extract(fit,"yhat")$yhat,probs=c(.1,.9))
within80 = yhat.ui80[,1] <= y & yhat.ui80[,2] >= y

yhat.ui50 = colQuantiles(extract(fit,"yhat")$yhat,probs=c(.25,.75))
within50 = yhat.ui50[,1] <= y & yhat.ui50[,2] >= y
lp = colMeans(extract(fit,"lp")$lp)

results = data.frame(m=opts$m,
                     train.mse=mean((yhat[1:n] - stan_data$y)^2),
           validate.mse=mean((yhat[(n+1):N] - heldout.y)^2),
           train.r2 = cor((yhat[1:n]), stan_data$y)^2,
           validate.r2=cor((yhat[(n+1):N]),heldout.y)^2,
           true.lengthscale=lengthscale,
           train.lp=mean(lp[1:n]),
           validate.lp=mean(lp[(n+1):N]),
           l=opts$l,
           k=opts$k,
           s=opts$s,
           d=opts$d,
           g=opts$g,
           S=opts$S,
           RandKcorrected="agree with python",
	   SDtoVARcorrected="true",
           train.coverage95=mean(within95[1:n]),
           validate.coverage95=mean(within95[(n+1):N]),
           train.coverage80=mean(within80[1:n]),
           validate.coverage80=mean(within80[(n+1):N]),
           train.coverage50=mean(within50[1:n]),
           validate.coverage50=mean(within50[(n+1):N]),
           iter=opts$i,
           elapsed=elapsed)

options(width=200)
print(t(results))

job_id = Sys.getenv("SLURM_JOB_ID")
if(job_id == "")
        job_id = Sys.getenv("PBS_JOBID")
hostname = Sys.info()["nodename"]
if(grepl("ziz",hostname)) { 
  path_to_output = "/data/ziz/not-backed-up/flaxman"
} else {
  path_to_output = "/work/sflaxman/bdr-data"
}

opts$d = basename(opts$d)
if(opts$T) {
  write.csv(results,sprintf("%s/test-results/results-%s-jobid:%s.csv",path_to_output,opts.str,job_id),row.names=F)
} else {
  cat(commandArgs(T),file=sprintf("%s/validate-results/results-jobid:%s.txt",path_to_output,job_id))
  write.csv(results,sprintf("%s/validate-results/results-%s-jobid:%s.csv",path_to_output,opts.str,job_id),row.names=F)
}

out=extract(fit)
fname = sprintf("%s/save-results/results-%s-jobid:%s", path_to_output,opts.str,job_id)

#save(fit,out,stan_data,train.y,test.y,validate.y,file=paste0(fname,".rdata"),row.names=F)

yhat.mu = colMeans(out$yhat)
yhat.sd = colSds(out$yhat)

write.csv(data.frame(y=stan_data$ytrue,mu=yhat.mu,sd=yhat.sd,lp=colMeans(out$lp)),paste0(fname,".csv"),row.names=F)
print(sprintf("Output is in %s",fname))

# 
# if(F) {
#   load("save-results/results-bdr-landmarks-blr-l_1-s_0.010000-k_30-g_false-d_chi2-manual-25-
#   load("save-results/results-bdr-landmarks-conjugacy-l_1-s_0.010000-k_30-g_true-d_chi2-manual-25-
#   load("save-results/results-bdr-landmarks-shrinkage-l_1-s_0.010000-k_30-g_true-d_chi2-manual-25-
#   yhat = colMeans(out$yhat)
#   n = stan_data$ntrain
#   N = stan_data$p
#   chk = data.table( cbind(yhat=yhat[(n+1):N],heldout.y,n=table(validate$V6)))
#   chk[, list(mse=mean((yhat-heldout.y)^2),r2=cor(yhat,heldout.y)^2),by=n]
#   apply(out$yhat[,(n+1):N][,chk$n == 1],1,sd)
# }
