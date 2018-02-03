data {
  int d; // dimensionality of the observed data
  int p; // number of bags
  int ntrain; // 1 ... ntrain are for training and ntrain+1 ... p are for testing
  
  matrix[p,d] mu;
  matrix[d,d] Sigma[p];
  vector[ntrain] y; // labels
  vector[p] ytrue; // labels (train+test)
}
parameters {
  vector[d] beta;

  real<lower=0> sigma;
  real alpha;
  real<lower=0> kappa;
}
transformed parameters {
  vector[p] mus;
  vector[p] sds;
  
  for(j in 1:p) {
    mus[j] = alpha + mu[j] * beta;
    sds[j] = sqrt(quad_form(Sigma[j],beta) + sigma); 
  }
}
model {
  for(j in 1:ntrain)
      y[j] ~ normal(mus[j],sds[j]); 

  alpha ~ normal(0,2);
  beta ~ normal(0,kappa);
  kappa ~ normal(0,2);
  sigma ~ normal(0,2);
}
generated quantities {
  vector[p] yhat;
  vector[p] lp;
  for(j in 1:p) {
    yhat[j] = normal_rng(mus[j],sds[j]);
    lp[j] = normal_lpdf(ytrue[j] | mus[j],sds[j]);
  }
}

