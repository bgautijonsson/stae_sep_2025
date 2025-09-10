// Data block: Defines all input data required by the model
data {
  int<lower=0> K;                     // Number of teams
  int<lower=0> N;                     // Number of games
  int<lower=0> N_rounds;              // Number of rounds
  int<lower=0> N_years;              // Number of years
  array[N] int<lower=1, upper=K> team1;    // Team 1 ID for each game (home)
  array[N] int<lower=1, upper=K> team2;    // Team 2 ID for each game (away)
  array[N] int<lower=1, upper = N_rounds> round1;  // Round ID for each game
  array[N] int<lower=1, upper = N_rounds> round2;  // Round ID for each game
  array[N] int<lower=1, upper=N_years> year; // Season ID for each game
  vector[N] goals1;                   // Goals scored by team 1 (home)
  vector[N] goals2;                   // Goals scored by team 2 (away)
  matrix[K, N_rounds] time_between_matches;  // Time difference between matches for each team and each round

  // Prediction data
  int<lower = 0> N_top_teams;
  array[N_top_teams] int<lower=0> top_teams;
  vector[N_top_teams] time_to_next_games;
  
  int<lower=0> N_pred;                // Number of games to predict
  array[N_pred] int<lower=1, upper=K> team1_pred;  // Team 1 ID for each prediction game
  array[N_pred] int<lower=1, upper=K> team2_pred;  // Team 2 ID for each prediction game
  vector[N_pred] pred_timediff1;
  vector[N_pred] pred_timediff2;
}

transformed data {
  matrix[K, N_rounds] delta_t;
  for (k in 1:K) {
    for (n in 1:N_rounds) {
      delta_t[k, n] = sqrt(time_between_matches[k, n]);
    }
  }

  vector[N_top_teams] delta_t_top = sqrt(time_to_next_games);
  vector[N_pred] pred_delta_t1 = sqrt(pred_timediff1);
  vector[N_pred] pred_delta_t2 = sqrt(pred_timediff2);
}

// Parameters to be estimated by the model
parameters {
  // Offensive parameters with hierarchical structure
  sum_to_zero_vector[K] off0;              // Initial offensive strengths
  array[N_rounds] sum_to_zero_vector[K] z_off;  // Innovations for offense
  vector[K] z_phi_off;                     // Team-specific AR(1) parameters
  real<lower=0> scale_phi_off;             // Population scale for AR(1)
  real mean_phi_off;                       // Population mean for AR(1)

  vector[K] z_theta_off; 
  real<lower = 0> scale_theta_off;
  real mean_theta_off;
  
  array[N_rounds] sum_to_zero_vector[K] z_mu_off;   // Random walk in mean offense
  
  vector[K] z_sigma_mu_off;         // Scale of random walk in mean offense
  real<lower = 0> scale_sigma_mu_off;
  real mean_sigma_mu_off;

  vector<lower = 0>[K] z_sigma_off;            // Scale of ARMA process
  real<lower = 0> scale_sigma_off;
  real mean_sigma_off;

  // Defensive parameters with hierarchical structure
  sum_to_zero_vector[K] def0;              
  array[N_rounds] sum_to_zero_vector[K] z_def;  

  vector[K] z_phi_def;   
  real<lower = 0> scale_phi_def;
  real mean_phi_def;

  vector[K] z_theta_def; 
  real<lower = 0> scale_theta_def;
  real mean_theta_def;

  array[N_rounds] sum_to_zero_vector[K] z_mu_def;   
  vector<lower = 0>[K] z_sigma_mu_def;   
  real<lower = 0> scale_sigma_mu_def;
  real mean_sigma_mu_def;
  
  vector<lower = 0>[K] z_sigma_def;            
  real<lower = 0> scale_sigma_def;
  real mean_sigma_def;

  // Team-specific sigma parameters
  vector[K] z_sigma_team;        // Team-specific scoring variability
  real<lower = 0> scale_sigma_team;
  real mean_sigma_team;
  
  // Home advantage parameters
  vector[K] home_advantage_off;            // Team-specific home offensive advantage
  vector[K] home_advantage_def;            // Team-specific home defensive advantage
  
  // Degrees of freedom for t-distribution
  real<lower = 1> nu;                      // Degrees of freedom for t-distribution
  
  // Correlation between home and away goals
  real<lower=-1, upper=1> rho;            // Correlation between home and away goals
  
  // Mean goals scored per game for each season
  real mu_goals_init;                      // Initial mean goals level
  vector[N_years - 1] z_mu_goals;          // Year-to-year changes in mean goals
  real<lower=0> sigma_mu_goals;            // Volatility of mean goals changes

  vector[N_pred] z_off_pred;
  vector[N_pred] z_def_pred;

  vector[N_top_teams] z_off_top;
  vector[N_top_teams] z_def_top;
}

// Transformed parameters: Derived quantities used in the model
transformed parameters {
  // Offensive parameters over time
  array[N_rounds] vector[K] offense;        // Offensive strengths for each round
  array[N_rounds] vector[K] mu_off;         // Mean offensive strengths for each round
  vector<lower = 0>[K] sigma_mu_off = exp(mean_sigma_mu_off + z_sigma_mu_off * scale_sigma_mu_off);
  vector<lower = 0>[K] sigma_off = exp(mean_sigma_off + z_sigma_off * scale_sigma_off);

  vector<lower = -1, upper = 1>[K] phi_off = 2 * inv_logit(mean_phi_off + z_phi_off * scale_phi_off) - 1;
  vector<lower = -1, upper = 1>[K] theta_off = 2 * inv_logit(mean_theta_off + z_theta_off * scale_theta_off) - 1;

  // Defensive parameters over time
  array[N_rounds] vector[K] defense;        // Defensive strengths for each round
  array[N_rounds] vector[K] mu_def;         // Mean defensive strengths for each round
  vector<lower = 0>[K] sigma_mu_def = exp(mean_sigma_mu_def + z_sigma_mu_def * scale_sigma_mu_def);
  vector<lower = 0>[K] sigma_def = exp(mean_sigma_def + z_sigma_def * scale_sigma_def);
  
  vector<lower = -1, upper = 1>[K] phi_def = 2 * inv_logit(mean_phi_def + z_phi_def * scale_phi_def) - 1;
  vector<lower = -1, upper = 1>[K] theta_def = 2 * inv_logit(mean_theta_def + z_theta_def * scale_theta_def) - 1;

  // Team-specific sigmas
  vector<lower=0>[K] sigma_team = exp(mean_sigma_team + scale_sigma_team * z_sigma_team);
  
  // Correlation matrix
  matrix[2,2] Omega;
  Omega[1,1] = 1;
  Omega[2,2] = 1;
  Omega[1,2] = rho;
  Omega[2,1] = rho;

  // Initialize first round
  mu_off[1] = z_mu_off[1];
  mu_def[1] = z_mu_def[1];
  offense[1, ] = mu_off[1] + off0;
  defense[1, ] = mu_def[1] + def0;

  // Second round combines AR, MA and noise components
  mu_off[2] = mu_off[1] + delta_t[ , 2] .* z_mu_off[2] .* sigma_mu_off;
  mu_def[2] = mu_def[1] + delta_t[ , 2] .* z_mu_def[2] .* sigma_mu_def;

  offense[2, ] = mu_off[2] +
    phi_off .* (offense[1, ] - mu_off[1]) +
    theta_off .* delta_t[ , 1] .* sigma_off .* z_off[1, ] +
    delta_t[ , 2] .* sigma_off .* z_off[2, ];

  defense[2, ] = mu_def[2] +
    phi_def .* (defense[1, ] - mu_def[1]) +
    theta_def .* delta_t[ , 1] .* sigma_def .* z_def[1, ] +
    delta_t[ , 2] .* sigma_def .* z_def[2, ];

  // Remaining rounds follow ARMA(1,1) process
  for (i in 3:N_rounds) {
    mu_off[i] = mu_off[i - 1] + delta_t[ , i] .* z_mu_off[i] .* sigma_mu_off;
    mu_def[i] = mu_def[i - 1] + delta_t[ , i] .* z_mu_def[i] .* sigma_mu_def;

    offense[i, ] = mu_off[i] +
      phi_off .* (offense[i - 1, ] - mu_off[i - 1]) +
      theta_off .* delta_t[ , i - 1] .* sigma_off .* z_off[i - 1, ] +
      delta_t[ , i] .* sigma_off .* z_off[i, ];

    defense[i, ] = mu_def[i] +
      phi_def .* (defense[i - 1, ] - mu_def[i - 1]) +
      theta_def .* delta_t[ , i - 1] .* sigma_def .* z_def[i - 1, ] +
      delta_t[ , i] .* sigma_def .* z_def[i, ];
  }

  // Mean goals for each season
  vector[N_years] mu_goals;
  // Initialize first year
  mu_goals[1] = mu_goals_init;
  // Random walk for mean goals
  for (i in 2:N_years) {
    mu_goals[i] = mu_goals[i-1] + z_mu_goals[i-1] * sigma_mu_goals;
  }
}

// Model block: Defines the likelihood and priors
model {
  // Priors for offensive parameters
  off0 ~ std_normal();
  for (i in 1:N_rounds) {
    z_off[i] ~ std_normal();
  }
  
  // Priors for defensive parameters
  def0 ~ std_normal();
  for (i in 1:N_rounds) {
    z_def[i] ~ std_normal();
  }
  
  // Priors for volatility parameters
  z_sigma_off ~ std_normal();
  scale_sigma_off ~ exponential(1);
  mean_sigma_off ~ normal(log(0.1), 1);

  z_sigma_mu_off ~ std_normal();
  scale_sigma_mu_off ~ exponential(1);
  mean_sigma_mu_off ~ normal(log(0.01), 1);

  z_sigma_def ~ std_normal();
  scale_sigma_def ~ exponential(1);
  mean_sigma_def ~ normal(log(0.1), 1);

  z_sigma_mu_def ~ std_normal();
  scale_sigma_mu_def ~ exponential(1);
  mean_sigma_mu_def ~ normal(log(0.01), 1);

  // Priors for ARMA parameters
  z_phi_off ~ std_normal();
  scale_phi_off ~ exponential(1);
  mean_phi_off ~ normal(0, 1);

  z_theta_off ~ std_normal();
  scale_theta_off ~ exponential(1);
  mean_theta_off ~ normal(0, 1);

  z_phi_def ~ std_normal();
  scale_phi_def ~ exponential(1);
  mean_phi_def ~ normal(0, 1);

  z_theta_def ~ std_normal();
  scale_theta_def ~ exponential(1);
  mean_theta_def ~ normal(0, 1);
  
  // Priors for random walk in mean
  for (i in 1:N_rounds) {
    z_mu_off[i] ~ std_normal();
    z_mu_def[i] ~ std_normal();
  }

  z_off_top ~ std_normal();
  z_def_top ~ std_normal();
  z_off_pred ~ std_normal();
  z_def_pred ~ std_normal();
  
  // Priors for home advantage components
  home_advantage_off ~ student_t(3, 0, 2);
  home_advantage_def ~ student_t(3, 0, 2);
  
  
  // Priors for scale, shape and correlation
  rho ~ uniform(-1, 1);
  nu ~ gamma(3, 0.15);

  // Priors for mean goals parameters
  mu_goals_init ~ normal(25, 5);           // Prior for initial mean goals
  z_mu_goals ~ std_normal();               // Standardized changes
  sigma_mu_goals ~ exponential(2);         // Prior for volatility of changes

  // Modify likelihood
  for (n in 1:N) {
    vector[2] off;
    vector[2] def;
    vector[2] mu;
    matrix[2,2] Sigma;
    // Home team
    off[1] = offense[round1[n], team1[n]] + // Base offensive strength
      home_advantage_off[team1[n]]; // Home advantage

    
    def[1] = defense[round1[n], team1[n]] + // Base defensive strength
      home_advantage_def[team1[n]]; // Home advantage


    // Away team
    off[2] = offense[round2[n], team2[n]]; // Base offensive strength

    def[2] = defense[round2[n], team2[n]]; // Base defensive strength


    mu[1] = mu_goals[year[n]] +                  // Season-specific base level
            (off[1] - def[2]); // Difference between offensive and defensive strengths
            
    mu[2] = mu_goals[year[n]] +                  // Season-specific base level
            (off[2] - def[1]);  // Difference between offensive and defensive strengths
            
    // Create game-specific covariance matrix using team sigmas
    Sigma[1,1] = square(sigma_team[team1[n]]);
    Sigma[2,2] = square(sigma_team[team2[n]]);
    Sigma[1,2] = rho * sigma_team[team1[n]] * sigma_team[team2[n]];
    Sigma[2,1] = Sigma[1,2];
    
    [goals1[n], goals2[n]]' ~ multi_student_t(nu, mu, Sigma);
  }
}

// Generated quantities: Post-estimation predictions and simulations
generated quantities {
  // Total home advantage
vector[K] home_advantage = home_advantage_off + home_advantage_def;

// Current team strengths accounting for recent performance
vector[N_top_teams] cur_offense_away = mu_off[N_rounds, top_teams] +
  phi_off[top_teams] .* (offense[N_rounds, top_teams] - mu_off[N_rounds, top_teams]) +
  theta_off[top_teams] .* delta_t_top .* sigma_off[top_teams] .* z_off[N_rounds, top_teams] +
  z_off_top .* sigma_off[top_teams]; 

vector[N_top_teams] cur_defense_away = defense[N_rounds, top_teams] +
  phi_def[top_teams] .* (defense[N_rounds, top_teams] - mu_def[N_rounds, top_teams]) +
  theta_def[top_teams] .* delta_t_top .* sigma_def[top_teams] .* z_def[N_rounds, top_teams] +
  z_def_top .* sigma_def[top_teams];

  vector[N_top_teams] cur_strength_away = cur_offense_away + cur_defense_away;

  // Current team strengths on home field accounting for recent performance
  vector[N_top_teams] cur_offense_home = cur_offense_away + home_advantage_off[top_teams]; 
  vector[N_top_teams] cur_defense_home = cur_defense_away + home_advantage_def[top_teams];
  vector[N_top_teams] cur_strength_home = cur_offense_home + cur_defense_home;

  vector[N_top_teams] cur_offense = (cur_offense_away + cur_offense_home) / 2;
  vector[N_top_teams] cur_defense = (cur_defense_away + cur_defense_home) / 2;
  vector[N_top_teams] cur_strength = cur_offense + cur_defense;
  
  vector[N_pred] goals1_pred;          // Predicted goals for team 1
  vector[N_pred] goals2_pred;          // Predicted goals for team 2
  vector[N_pred] goal_diff_pred;       // Predicted goal difference
  vector[N_pred] total_goals_pred;     // Predicted total goals
  
  // Replicated data for model checking
  vector[N] goals1_rep;                         
  vector[N] goals2_rep;                         
  vector[N] goal_diff_rep;
  vector[N] total_goals_rep;
  
  // Generate predictions for future games
  for (n in 1:N_pred) {
    vector[2] off;
    vector[2] def;
    vector[2] mu;
    matrix[2,2] Sigma;
    
    // Home team
    off[1] = mu_off[N_rounds, team1_pred[n]] +
      phi_off[team1_pred[n]] .* (offense[N_rounds, team1_pred[n]] - mu_off[N_rounds, team1_pred[n]]) +
      theta_off[team1_pred[n]] .* pred_delta_t1[n] .* sigma_off[team1_pred[n]] .* z_off[N_rounds, team1_pred[n]] +
      z_off_pred[n] .* sigma_off[team1_pred[n]];

    def[1] = mu_def[N_rounds, team1_pred[n]] +
      phi_def[team1_pred[n]] .* (defense[N_rounds, team1_pred[n]] - mu_def[N_rounds, team1_pred[n]]) +
      theta_def[team1_pred[n]] .* pred_delta_t1[n] .* sigma_def[team1_pred[n]] .* z_def[N_rounds, team1_pred[n]] +
      z_def_pred[n] .* sigma_def[team1_pred[n]];



    // Away team
    off[2] = mu_off[N_rounds, team2_pred[n]] +
      phi_off[team2_pred[n]] .* (offense[N_rounds, team2_pred[n]] - mu_off[N_rounds, team2_pred[n]]) +
      theta_off[team2_pred[n]] .* pred_delta_t2[n] .* sigma_off[team2_pred[n]] .* z_off[N_rounds, team2_pred[n]] +
      z_off_pred[n] .* sigma_off[team2_pred[n]];

    def[2] = mu_def[N_rounds, team2_pred[n]] +
      phi_def[team2_pred[n]] .* (defense[N_rounds, team2_pred[n]] - mu_def[N_rounds, team2_pred[n]]) +
      theta_def[team2_pred[n]] .* pred_delta_t2[n] .* sigma_def[team2_pred[n]] .* z_def[N_rounds, team2_pred[n]] +
      z_def_pred[n] .* sigma_def[team2_pred[n]];



    mu[1] = mu_goals[year[n]] +                  // Use most recent season's base level
            (off[1] - def[2]); // Difference between offensive and defensive strengths

    mu[2] = mu_goals[year[n]] +                  // Use most recent season's base level
            (off[2] - def[1]); // Difference between offensive and defensive strengths
            
    // Use team-specific sigmas for predictions
    Sigma[1,1] = square(sigma_team[team1_pred[n]]);
    Sigma[2,2] = square(sigma_team[team2_pred[n]]);
    Sigma[1,2] = rho * sigma_team[team1_pred[n]] * sigma_team[team2_pred[n]];
    Sigma[2,1] = Sigma[1,2];
    
    vector[2] y = multi_student_t_rng(nu, mu, Sigma);
    goals1_pred[n] = y[1];
    goals2_pred[n] = y[2];
    goal_diff_pred[n] = y[1] - y[2];
    total_goals_pred[n] = y[1] + y[2];
  }

  // Generate replicated data for model checking
  for (n in 1:N) {
    vector[2] off;
    vector[2] def;
    vector[2] mu;
    matrix[2,2] Sigma;
    // Home team
    off[1] = offense[round1[n], team1[n]] + // Base offensive strength
      home_advantage_off[team1[n]]; // Effect of goals scored

    def[1] = defense[round1[n], team1[n]] + // Base defensive strength
      home_advantage_def[team1[n]]; // Home advantage


    // Away team

    // Away team
    off[2] = offense[round2[n], team2[n]]; // Base offensive strength

    def[2] = defense[round2[n], team2[n]]; // Base defensive strength
            
    mu[1] = mu_goals[year[n]] +                  // Season-specific base level
            (off[1] - def[2]); // Difference between offensive and defensive strengths
            
    mu[2] = mu_goals[year[n]] +                  // Season-specific base level
            (off[2] - def[1]); // Difference between offensive and defensive strengths
            
    // Use team-specific sigmas for replications
    Sigma[1,1] = square(sigma_team[team1[n]]);
    Sigma[2,2] = square(sigma_team[team2[n]]);
    Sigma[1,2] = rho * sigma_team[team1[n]] * sigma_team[team2[n]];
    Sigma[2,1] = Sigma[1,2];
    
    vector[2] y = multi_student_t_rng(nu, mu, Sigma);
    goals1_rep[n] = y[1];
    goals2_rep[n] = y[2];
    goal_diff_rep[n] = y[1] - y[2];
    total_goals_rep[n] = y[1] + y[2];
  }
}

