library(data.table)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

train = fread("../data/chi2_23/train_x.csv",header=F)
test = fread("../data/chi2_23/test_x.csv",header=F)
validate = fread("../data/chi2_23/val_x.csv",header=F)
train.y = read.csv("../data/chi2_23/train_y.csv",header=F)[,1]
test.y = read.csv("../data/chi2_23/test_y.csv",header=F)[,1]
validate.y = read.csv("../data/chi2_23/val_y.csv",header=F)[,1]
d = ncol(train)-1

median.bw = median(as.numeric(as.matrix(dist(train[sample(nrow(train[,1:d]),1000),1:d]))))

K = 45
freq = matrix(rnorm(K*d),d)
X = rbind(data.matrix(train[,1:d]),data.matrix(validate[,1:d]))
phi = cbind(cos(X %*% freq/median.bw),sin(X %*% freq/median.bw)) / sqrt(2*K)

bag_index = c(as.integer(unlist(train[,d+1,with=F]))+1,as.integer(unlist(validate[,d+1,with=F])) + length(train.y) + 1)
stan_data = list(d = ncol(phi),
                 n = nrow(train)+nrow(validate),
                 p = length(train.y) + length(validate.y),
                 ntrain=length(train.y),
                 bag_index = bag_index,
                 X = phi,
                 y=train.y)
m = stan_model("bdr1.stan")

fit = sampling(m,data=stan_data,iter=400,warmup=200,chains=4)

print(fit,"beta")

print(fit,c("gamma","delta","sigma"))
yhat = colMeans(extract(fit,"yhat")$yhat)
print(sprintf("train MSE = %.03f",mean((yhat[1:length(train.y)] - train.y)^2)))
print(sprintf("validate MSE = %.03f",mean((yhat[(length(train.y)+1):(length(train.y)+length(validate.y))] - validate.y)^2)))
