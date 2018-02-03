data {
  int d; // dimensionality of the observed data
  int n; // number of samples
  int p; // number of bags
  int ntrain; // 1 ... ntrain are for training and ntrain+1 ... p are for testing
  
  int bag_index[n]; // entries should be in [1..p]
//  matrix[n,d] X; // samples
  matrix[p,d] mu;
  vector[ntrain] y; // labels
  matrix[d,d] R;
  vector[p] ytrue; // labels (train+test)
}
parameters {
  vector[d] beta;

  real<lower=0> sigma;
  real<lower=0> kappa;
  real alpha;
}
model {
  for(j in 1:ntrain)
      y[j] ~ normal(alpha + mu[j] * beta,sigma);

  alpha ~ normal(0,2);
  beta ~ normal(0,kappa);
  kappa ~ normal(0,2);
  sigma ~ normal(1,2);
}
generated quantities {
  vector[p] yhat;
  vector[p] lp;
  for(k in 1:p) {
    yhat[k] = normal_rng(alpha + mu[k] * beta,sigma);
    lp[k] = normal_lpdf(ytrue[k] | alpha + mu[k] * beta,sigma);
  }
}
