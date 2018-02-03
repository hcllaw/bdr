NB. THE MATH HERE ISN'T QUITE RIGHT

data {
  int d; // dimensionality of the observed data
  int n; // number of samples
  int p; // number of bags
  int ntrain; // 1 ... ntrain are for training and ntrain+1 ... p are for testing
  
  int bag_index[n]; // entries should be in [1..p]
  matrix[n,d] X; // samples
  matrix[p,d] mu;
  matrix[d,d] Sigma[p];
  vector[ntrain] y; // labels
  vector[p] ytrue; // labels (train+test)
  matrix[d,d] R;
}
transformed data {
  matrix[d,d] L;
  vector[d] zeros;
  zeros = rep_vector(0,d);
  L = cholesky_decompose(R);
}
parameters {
  vector[d] beta;
  vector[d] mu0;

  real<lower=0> sigma;
  real alpha;
  real<lower=0> kappa;
}
transformed parameters {
  vector[p] mus;
  vector[p] sds;
  real mu0beta;
  
  mu0beta = transpose(mu0) * beta;
  for(j in 1:p) {
    mus[j] = alpha + mu[j] * beta + mu0beta;
    sds[j] = quad_form(Sigma[j],beta) + sigma; 
  }
}
model {
  for(j in 1:ntrain)
      y[j] ~ normal(mus[j],sds[j]); 
  mu0 ~ multi_normal_cholesky(zeros,L);
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

