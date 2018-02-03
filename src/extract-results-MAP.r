library(rstan)
library(matrixStats)
args = commandArgs(trailingOnly=TRUE)
fnames = NULL
for(fname in args) {
 load(fname)
  print(sprintf("loading %s",fname))
  fname = sprintf("%s.csv",strsplit(fname,".rdata")[[1]])

  lp = map$par[grep("^lp",names(map$par))]
  yhat.mu = map$par["alpha"] + stan_data$mu %*% map$par[grep("beta",names(map$par))]
  yhat.sd = map$par[grep("sds",names(map$par))]

  
  write.csv(data.frame(y=stan_data$ytrue,mu=yhat.mu,sd=yhat.sd,lp=lp),fname,row.names=F)
  print(sprintf("Output is in %s",fname))
fnames = c(fnames,fname)
}
cat("zip archive.zip", paste(fnames), "\n")


