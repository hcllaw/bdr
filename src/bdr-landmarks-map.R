library(docopt)

'Usage:
   bdr-landmarks.R [-l lengthscale] [-s sigma] [-k landmarks] [-d dataset] [-i iter] [-r recompile] [-T test] [-m bdr-landmarks-conjugacy] [-g global-mean]  [-S standardize]

Options:
   -l lengthscale [default: 1]
   -s sigma [default: 1]
   -k landmarks [default: 2]
   -d dataset [default: chi2-manual-25]
   -r recompile [default: false]
   -T include testing data? [default: false]
   -m model name [default: bdr-landmarks-blr]
   -g shrink towards a global mean? [default: true]
   -S standardize labels [default: false]
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
table(table(train$V6))

test = fread(sprintf("../data/%s/test_x.csv",opts$d),header=F)
validate = fread(sprintf("../data/%s/val_x.csv",opts$d),header=F)
train.y = read.csv(sprintf("../data/%s/train_y.csv",opts$d),header=F)[,1]
test.y = read.csv(sprintf("../data/%s/test_y.csv",opts$d),header=F)[,1]
validate.y = read.csv(sprintf("../data/%s/val_y.csv",opts$d),header=F)[,1]

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
  u = data.matrix(fread(sprintf("../data/%s/landmarks-%d.csv",opts$d,opts$k),header=F))
  print("successfully read in landmark points")
}, error = function(e) {
  print("randomly chose landmark points")
})

R = kernelMatrix(rbfdot(1 / (sqrt(2)*lengthscale)^2),u) + diag(1e-6,nrow(u)) # add some jitter to preserve positive definiteness
phi = kernelMatrix(rbfdot(1 / lengthscale^2),X,u)

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
                 n = nrow(phi),
                 p = p,
                 ntrain=length(train.y), 
                 bag_index = bag_index,
                 X = phi,
                 mu=mu,
                 y=train.y,
                 ytrue=c(train.y,validate.y))

if(opts$g == "true") { # use a global mean
  mu0 = colMeans(phi)
} else {
  mu0 = rep(0,opts$k)
}
if(opts$m == "bdr-landmarks-conjugacy" | opts$m == "bdr-landmarks-shrinkage") {
  Sigma = array(0, dim=c(p,d,d))
  
  for(i in 1:p) {
    bag = phi[bag_index == i,]
    n = nrow(bag)
    if(is.null(n)) # just one row
      n = 1
    
    mu[i,] = R %*% solve(R+diag(opts$s/n,d),mu[i,] - mu0) + mu0
    Sigma[i,,] = R - R %*% solve(R+diag(opts$s/n,d),R)
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

ptm = proc.time()
map = optimizing(m,data=stan_data,hessian=T)

elapsed = as.numeric((proc.time() - ptm)[3])

yhat = map$par["alpha"] + stan_data$mu %*% map$par[grep("beta",names(map$par))]
if("sds" %in% names(map$par)) {
  yhat.sd = map$par[grep("sds",names(map$par))]
} else {
  yhat.sd = map$par[grep("sigma",names(map$par))]
}

n = stan_data$ntrain
N = stan_data$p

# check calibration
library(matrixStats)
y = c(stan_data$y,heldout.y)
lp = map$par[grep("^lp",names(map$par))]

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
           RandKcorrected="agree with python",
           S=opts$S,
           d=opts$d,
           g=opts$g,
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
  write.csv(results,sprintf("%s/test-results/results-MAP-%s-l_%s-s_%f-k_%d-g_%s-d_%s-j_%s.csv",path_to_output,opts$m,opts$l,opts$s,opts$k,opts$g,opts$d,job_id),row.names=F)
} else {
  write.csv(results,sprintf("%s/validate-results/results-MAP-%s-l_%s-s_%f-k_%d-g_%s-d_%s-j_%s.csv",path_to_output,opts$m,opts$l,opts$s,opts$k,opts$g,opts$d,job_id),row.names=F)
}

out = sprintf("%s/save-results/results-MAP-%s-l_%s-s_%f-k_%d-g_%s-d_%s-j_%s.csv",path_to_output,opts$m,opts$l,opts$s,opts$k,opts$g,opts$d,job_id)
write.csv(data.frame(y=y,mu=yhat,sd=yhat.sd,lp=lp),out,row.names=F)

save(map,stan_data,train.y,test.y,validate.y,file=sprintf("%s/save-results/results-MAP-%s-l_%s-s_%f-k_%d-g_%s-d_%s-j_%s.rdata",path_to_output,opts$m,opts$l,opts$s,opts$k,opts$g,opts$d,job_id),row.names=F)
