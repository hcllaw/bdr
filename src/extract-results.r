library(rstan)
library(matrixStats)
args = commandArgs(trailingOnly=TRUE)
fnames = NULL
for(fname in args) {
 load(fname)
  print(sprintf("loading %s",fname))
  yhat.mu = colMeans(out$yhat)
  yhat.sd = colSds(out$yhat)
  fname = sprintf("%s.csv",strsplit(fname,".rdata")[[1]])
  
  write.csv(data.frame(y=stan_data$ytrue,mu=yhat.mu,sd=yhat.sd,lp=colMeans(out$lp)),fname,row.names=F)
  print(sprintf("Output is in %s",fname))
fnames = c(fnames,fname)
}
cat("zip archive.zip", paste(fnames), "\n")


