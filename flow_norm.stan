//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//

// The input data is a vector 'y' of length 'n_obs'.
data {
  int<lower=0> n_obs; // n_obsumber of observations
  vector[n_obs] C;
  vector[n_obs] y;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  // intercepts
  real intercept; // Global of lp (int)
  real<lower=0> sigma;
  real beta_Q; // flow effects (eff.Q)  
}

transformed parameters{
  vector[n_obs] mu;
  
  // likelihood:
  for (i in 1:n_obs){
    mu[i] = intercept + beta_Q*C[i];
  }
}

model {
  intercept ~ std_normal();
  beta_Q ~ std_normal();
  sigma ~ std_normal();
  
  for (i in 1:n_obs){
    target += normal_lpdf(y[i] | mu[i], sigma);
  }
}
generated quantities{
  vector[n_obs] y_pred;  // Posterior predictive samples
  vector[n_obs] log_lik;
  for (i in 1:n_obs) {
    y_pred[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
}

