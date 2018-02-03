data {
  int d; // dimensionality of the observed data
  int n; // number of samples
  int p; // number of bags
  int ntrain; // 1 ... ntrain are for training and ntrain+1 ... p are for testing
  
  int bag_index[n]; // entries should be in [1..p]
  matrix[n,d] X; // samples
  vector[ntrain] y; // labels
  matrix[d,d] R;
}
transformed data {
  matrix[d,d] L;
  vector[d] zeros;
  zeros = rep_vector(0,d);
  L = cholesky_decompose(R);
}
parameters {
  matrix[p,d] mu;
  vector[d] z;
  vector[d] beta;

  real<lower=0> sigma;
  real<lower=0> gamma;
  real<lower=0> delta;
  real<lower=0> lambda;
  real alpha;
  real<lower=0> kappa;
}
transformed parameters {
  row_vector[d] mu0;
  mu0 = to_row_vector(L * z);
}
model {
  for(i in 1:n) {
    X[i] ~ normal(mu0 + mu[bag_index[i]],sigma);
  }
  for(j in 1:p) {
    mu[j] ~ normal(0,gamma); // change this to multi_normal_cholesky(zeros,L) ??
  }
  for(j in 1:ntrain)
      y[j] ~ normal(alpha + (mu0 + mu[j]) * beta,delta);

  z ~ normal(0,1);
  beta ~ normal(0,kappa);
  kappa ~ normal(0,2);
  sigma ~ normal(0,2);
  gamma ~ normal(0,2);
  lambda ~ normal(0,2);
  delta ~ normal(1,2);
  alpha ~ normal(0,5);
}
generated quantities {
  vector[p] yhat;
  for(k in 1:p)
    yhat[k] = normal_rng(alpha + (mu0 + mu[k]) * beta,delta);
}
