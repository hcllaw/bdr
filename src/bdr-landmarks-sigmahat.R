library(docopt)

'Usage:
   bdr-landmarks.R [-l lengthscale] [-s sigma] [-k landmarks] [-d dataset] [-r recompile] [-T test] [-m bdr-landmarks-conjugacy]

Options:
   -l lengthscale [default: 1]
   -s sigma [default: 1]
   -k landmarks [default: 2]
   -d dataset [default: chi2-varybagsize-50]
   -r recompile [default: true]
   -T include testing data? [default: false]
   -m model name [default: bdr-landmarks-conjugacy]
' -> doc

opts = docopt(doc)
opts$s = as.numeric(opts$s)
opts$k = as.numeric(opts$k)
opts$l = as.numeric(opts$l)
library(data.table)
library(rstan)
library(kernlab)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

print("Reading in data...")
train = fread(sprintf("../data/%s/train_x.csv",opts$d),header=F)
test = fread(sprintf("../data/%s/test_x.csv",opts$d),header=F)
validate = fread(sprintf("../data/%s/val_x.csv",opts$d),header=F)
train.y = read.csv(sprintf("../data/%s/train_y.csv",opts$d),header=F)[,1]
test.y = read.csv(sprintf("../data/%s/test_y.csv",opts$d),header=F)[,1]
validate.y = read.csv(sprintf("../data/%s/val_y.csv",opts$d),header=F)[,1]

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
R = kernelMatrix(rbfdot(.5 / (2*lengthscale)^2),u) + diag(1e-6,nrow(u)) # add some jitter to preserve positive definiteness
phi = kernelMatrix(rbfdot(.5 / lengthscale^2),X,u)

bag_index = c(as.integer(unlist(train[,d+1,with=F]))+1,
              as.integer(unlist(validate[,d+1,with=F])) + length(train.y) + 1)
if(opts$T)
  bag_index = c(bag_index,as.integer(unlist(test[,d+1,with=F])) + length(validate.y) + length(train.y) + 1)

mu = data.table(phi)[, lapply(.SD, mean), by=bag_index]
mu[, bag_index := NULL]

p = length(unique(bag_index))

d = ncol(phi)
Sigma = array(0, dim=c(p,d,d))
n = nrow(phi)
H = diag(n) - 1/n * matrix(1,nrow=n,ncol=n)

sigma.hat = 1/d * t(phi) %*% H %*% phi
print(dim(sigma.hat))
print(dim(R))
for(i in 1:p) {
  bag = phi[bag_index == i,]
  n = nrow(bag)
  if(is.null(n)) # just one row
    n = 1
  Sigma[i,,] = R - R %*% solve(R+sigma.hat/n,R)
}
stan_data = list(d = d,
                 n = nrow(phi),
                 p = p,
                 ntrain=length(train.y), 
                 bag_index = bag_index,
                 X = phi,
                 Sigma=Sigma,
                 mu=data.matrix(mu),
                 y=train.y,
                 ytrue=c(train.y,validate.y))
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

ptm = proc.time()
fit = sampling(m,data=stan_data,iter=200,warmup=100,chains=4)
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
           train.coverage95=mean(within95[1:n]),
           validate.coverage95=mean(within95[(n+1):N]),
           train.coverage80=mean(within80[1:n]),
           validate.coverage80=mean(within80[(n+1):N]),
           train.coverage50=mean(within50[1:n]),
           validate.coverage50=mean(within50[(n+1):N]),
           elapsed=elapsed)

options(width=200)
print(t(results))

job_id = Sys.getenv("SLURM_JOB_ID")
hostname = Sys.info()["nodename"]
if(grepl("ziz",hostname)) { 
  path_to_output = "/data/ziz/not-backed-up/flaxman"
} else {
  path_to_output = "."
}
if(opts$T) {
  write.csv(results,sprintf("%s/test-results/results-%s-l_%s-s_%f-k_%d-t_%s-j_%s.csv",path_to_output,opts$m,opts$l,opts$s,opts$k,opts$d,job_id),row.names=F)
} else {
  write.csv(results,sprintf("%s/validate-results/results-%s-l_%s-s_%f-k_%d-t_%s-j_%s.csv",path_to_output,opts$m,opts$l,opts$s,opts$k,opts$d,job_id),row.names=F)
}
