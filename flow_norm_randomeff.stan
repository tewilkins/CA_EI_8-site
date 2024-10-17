//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//

// The input data is a vector 'y' of length 'n_obs'.
data {
  int<lower=0> n_obs; // n_obsumber of observations
  int<lower=0> n_samp; // n_obsumber of samples per site (ordered downstream)
  int<lower=0> n_pos; // n_obsumber of positions (upstream, downstream or mixed)
  int<lower=0> n_habitat; // n_obsumber of sites
  int<lower=0> n_site; // n_obsumber of sites
  int<lower=0> samp_no[n_obs];
  int<lower=0> pos_no[n_obs];
  int<lower=0> habitat_no[n_obs];
  int<lower=0> site_no[n_obs];
  vector[n_obs] y;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  // intercepts
  real intercept; // Global of lp (int)
  real<lower=0> sigma;
  
  // sample effects
  vector[n_samp] alpha_samp_raw;
  real<lower=0> sigma_samp;
  
  // position effects
  vector[n_pos] alpha_pos_raw;
  real<lower=0> sigma_pos;  
  
  //  effects
  vector[n_habitat] alpha_habitat_raw;
  real<lower=0> sigma_habitat; 
  
  //  effects
  vector[n_site] alpha_site_raw;
  real<lower=0> sigma_site;   
}

transformed parameters{
  vector[n_samp] alpha_samp;             // Random effect of sample downstream
  vector[n_pos] alpha_pos;
  vector[n_habitat] alpha_habitat;
  vector[n_site] alpha_site;
  vector[n_obs] mu;
  
  alpha_samp[1] = alpha_samp_raw[1] * sigma_samp;
  for (m in 2:n_samp){
    alpha_samp[m] = alpha_samp[m-1] + alpha_samp_raw[m] * sigma_samp;
  }
  
  alpha_site[1] = alpha_site_raw[1] * sigma_site;
  for (s in 2:n_site){
    alpha_site[s] = alpha_site[s-1] + alpha_site_raw[s] * sigma_site;
  }
  
  for (h in 1:n_habitat){
    alpha_habitat[h] = alpha_habitat_raw[h] * sigma_habitat;
  }
  
  for (p in 1:n_pos){
    alpha_pos[p] = alpha_pos_raw[p] * sigma_pos;
  }
  
  // likelihood:
  for (i in 1:n_obs){
    mu[i] = intercept + alpha_samp[samp_no[i]] + alpha_pos[pos_no[i]] + alpha_habitat[habitat_no[i]] + alpha_site[site_no[i]];
  }
}

model {
  
  alpha_samp_raw ~ std_normal();
  sigma_samp ~ std_normal();
  
  alpha_habitat_raw ~ std_normal();
  sigma_habitat ~ std_normal();

  alpha_pos_raw ~ std_normal();
  sigma_pos ~ std_normal();
  
  alpha_site_raw ~ std_normal();
  sigma_site ~ std_normal();
  
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

