%% This script reproduces the results (Figure 2) of the simulation study 1, in the Results section.

clear all; close all; clc;

% Generating simulated data
[Y, alpha, mu_x, Sigma_w, noise_covariance_true, avg_spiking_rate, A, s, D_true, ground_truth_spikes] = generate_signals();

% Setting parameters
N = size(Y, 2); % number of neurons
K = size(Y, 1); % number of time frames
L = size(Y, 3); % number of independent realization
M = size(s,1); % number of time lags of the stimulus, taken into consideration

%% Hyper parameter tuning - round 1 - fixing psi_x and varying rho_x /gamma_x

psi_x_initial = eye(N); % we consider psi_x = psi_x_initial*eta
eta = 1000; % fix the scale matrix in round 1
gamma_x_range = [58000, 60000]; % candidate values for gamma_x (= rho_x + K)
D_hat_initial_estimate = zeros(N,M); % initial estimate of the weight matrix
error_threshold_gamma_x = 0.1;

% Initializing the variables
valid_result_gamma = zeros(1,length(gamma_x_range)); % this is to record whether the estimate is valid. Will be 1 if the estimate is not finite
cov_estimates_gamma = zeros(N, N, length(gamma_x_range)); % recording the covariance estimates at all parameter settings
D_hat_gamma =  zeros(N, M, length(gamma_x_range)); % recording the weight matrix (D) estimates at all parameter settings
z_k_k_gamma = zeros(N, K, L, length(gamma_x_range)); % recording the calcium concentration estimates at all parameter settings
n_hat_gamma = zeros(N, K, L, length(gamma_x_range)); % recording the spike estimates at all parameter settings
m_x_gamma = zeros(N, K, L, length(gamma_x_range)); % recording the latent noise process at all parameter settings
performance_eval_gamma = zeros(1,length(gamma_x_range)); % recording the disparity of covariance estimates at all parameter settings, in order to evaluate the optimal settings

% Deriving the proposed estimate for each hyper-parameter setting
for i = 1:length(gamma_x_range)
    [Sigma_x, z_k_k, n_hat, D_hat, m_x] = proposed_estimation_procedure(Y, Sigma_w, mu_x, gamma_x_range(i), eta*psi_x_initial, alpha, A, s, D_hat_initial_estimate,error_threshold_gamma_x);
    cov_estimates_gamma(:,:, i) = Sigma_x;
    D_hat_gamma(:, :, i) = D_hat;
    z_k_k_gamma(:,:,:,i) = z_k_k;
    n_hat_gamma(:,:,:,i) = n_hat;
    m_x_gamma(:,:,:,i) = m_x;
    if sum(sum(isnan(Sigma_x))) > 0 || sum(sum(isinf(Sigma_x))) > 0
        valid_result_gamma(i) = valid_result_gamma(i) + 1;
        continue
    else
    performance_eval_gamma(i) = performance_evaluation(Y, Sigma_x, Sigma_w, alpha, A, mu_x, D_hat, s); % evaluating the performance of the current estimate
    end
end

% Intitial proposed estimates
[~,selected_setting_initial] = min(performance_eval_gamma); % select the initial optimal estimate setting (with minimal disparity)
noise_covariance_initial_estimate = cov_estimates_gamma(:,:,selected_setting_initial); % initial noise covariance estimate
D_hat_initial_estimate = squeeze(squeeze(D_hat_gamma(:,:,selected_setting_initial))); % initial stimulus weight matrix estimate

%%  Hyper parameter tuning - round 2 - fixing gamma_x and varying psi_x by changing eta 

gamma_x = 100000; % fix gamma_x (= rho_x + KL) at an optimal setting in round 2
psi_x_initial = noise_covariance_initial_estimate; % use the covariance estimated from round 1 as the base of the scale matrix in round 2
eta_range = [1000,1200]; % candidate values for eta
error_threshold_psi = 0.001;

% Initializing the variables
valid_result_psi = zeros(1,length(eta_range)); % this is to record whether the estimate is valid. Will be 1 if the estimate is not finite
cov_estimates_psi = zeros(N, N, length(eta_range)); % recording the covariance estimates at all parameter settings
D_hat_psi =  zeros(N, M, length(eta_range)); % recording the weight matrix (D) estimates at all parameter settings
z_k_k_psi = zeros(N, K, L, length(eta_range)); % recording the calcium concentration estimates at all parameter settings
n_hat_psi = zeros(N, K, L, length(eta_range)); % recording the spike estimates at all parameter settings
m_x_psi = zeros(N, K, L, length(eta_range)); % recording the latent noise process at all parameter settings
performance_eval_psi = zeros(1,length(eta_range)); % recording the disparity of covariance estimates at all parameter settings, in order to evaluate the optimal settings

% Deriving the proposed estimate for each hyper-parameter setting
for i = 1:length(eta_range)
    [Sigma_x, z_k_k, n_hat, D_hat, m_x] = proposed_estimation_procedure(Y, Sigma_w, mu_x, gamma_x, eta_range(i)*psi_x_initial, alpha, A, s, D_hat_initial_estimate,error_threshold_psi); 
    cov_estimates_psi(:,:, i) = Sigma_x;
    D_hat_psi(:, :, i) = D_hat;
    z_k_k_psi(:,:,:,i) = z_k_k;
    n_hat_psi(:,:,:,i) = n_hat;
    m_x_psi(:,:,:,i) = m_x;
    if sum(sum(isnan(Sigma_x))) > 0 || sum(sum(isinf(Sigma_x))) > 0
        valid_result_psi(i) = valid_result_psi(i) + 1;
        continue
    else
    performance_eval_psi(i) = performance_evaluation(Y, Sigma_x, Sigma_w, alpha, A, mu_x, D_hat, s); % evaluating the performance of the current estimate
    end
end

% Final proposed estimates after parameter optimization
[~,selected_setting_final] = min(performance_eval_psi); % select the final optimal setting, that minimizes disparity
noise_covariance_Proposed = cov_estimates_psi(:,:,selected_setting_final); % final Proposed noise covariance estimate
D_hat_proposed = squeeze(squeeze(D_hat_psi(:,:,selected_setting_final))); % final stimulus weight matrix estimate

%% Plotting results of hyper paramter tuning
 
% Plotting the hyper parameter tuning results of round 1 - varying gamma_x
figure(3);
subplot(2,1,1);
plot(gamma_x_range, performance_eval_gamma);
title('The variation of disparity with \gamma_x: hyper parameter tuning round 1');
grid on;
axis([min(gamma_x_range), max(gamma_x_range), min(performance_eval_gamma), max(performance_eval_gamma)]);

% Plotting the hyper parameter tuning results of round 2 - varying eta
subplot(2,1,2);
plot(eta_range, performance_eval_psi);
title('The variation of disparity with \eta: hyper parameter tuning round 2');
grid on;
axis([min(eta_range), max(eta_range), min(performance_eval_psi), max(performance_eval_psi)]);

%% Two-Stage estimation: Spike estimation by FCSS method and smoothing the estimated spikes

estimated_spikes_FCSS =  spike_deconvolution_FCSS(Y, A);

%Filtering the estimated spikes with a Gaussian Kernel to get a continuous
%process
window_length = 50;
alpha_kernel = 5;
kernel_used = gausswin(window_length,alpha_kernel);
spikes_smoothed_FCSS = double(estimated_spikes_FCSS); % generating the smoothed spikes
for l=1:L
    for j = 1:N
        spikes_smoothed_FCSS(j,:,l) = conv(squeeze(squeeze(spikes_smoothed_FCSS(j,:,l))),kernel_used,'same');
    end
end 

%% Computing noise correlations from all methods and plotting results

% Ground truth noise correlations
noise_correlation_ground_truth = diag(1./sqrt(diag(noise_covariance_true)))*noise_covariance_true*diag(1./sqrt(diag(noise_covariance_true)));

% Proposed Estimates for noise correlation
noise_correlation_Proposed = diag(1./sqrt(diag(noise_covariance_Proposed)))*noise_covariance_Proposed*diag(1./sqrt(diag(noise_covariance_Proposed)));

% Pearson Estimate for noise correlations
noise_covariance_Pearson = zeros(N);
for l = 1:L
    noise_covariance_Pearson = noise_covariance_Pearson + cov(squeeze(Y(:,:,l) - mean(Y, 3)));
end
noise_covariance_Pearson = noise_covariance_Pearson/L;
noise_correlation_Pearson =  diag(1./sqrt(diag(noise_covariance_Pearson)))*noise_covariance_Pearson*diag(1./sqrt(diag(noise_covariance_Pearson)));

% Two-stage Estimate for noise correlations
noise_covariance_Two_Stage = zeros(N);
for l=1:L
    noise_covariance_Two_Stage = noise_covariance_Two_Stage + cov(squeeze(spikes_smoothed_FCSS(:,:,l))' - mean(spikes_smoothed_FCSS, 3)');
end 
noise_covariance_Two_Stage = noise_covariance_Two_Stage/L;
noise_correlation_Two_Stage =  diag(1./sqrt(diag(noise_covariance_Two_Stage)))*noise_covariance_Two_Stage*diag(1./sqrt(diag(noise_covariance_Two_Stage)));

% Plotting the results of noise correlation estimation
fig_range_noise = 0.3;
figure(4);
% Ground truth noise correlations
subplot(2,4,1);
temp = noise_correlation_ground_truth;
for j = 1:N
    temp(j,j)=0;
end
final_noise_correlation_ground_truth = temp / norm(temp,2);
imagesc(1:N, 1:N, final_noise_correlation_ground_truth, [-1*fig_range_noise, fig_range_noise]);
colorbar;
colormap(redblue);
title('Ground truth noise correlations', 'Interpreter', 'latex');
% Proposed noise correlation estimates
subplot(2,4,2);
temp = noise_correlation_Proposed;
for j = 1:N
    temp(j,j)=0;
end
final_noise_correlation_Proposed = temp / norm(temp,2);
imagesc(1:N, 1:N, final_noise_correlation_Proposed, [-1*fig_range_noise, fig_range_noise]);
colorbar;
colormap(redblue);
title('Proposed noise correlations', 'Interpreter', 'latex');
% Pearson noise correlation estimates
subplot(2,4,3);
temp = noise_correlation_Pearson;
for j = 1:N
    temp(j,j)=0;
end
final_noise_correlation_Pearson = temp / norm(temp,2);
imagesc(1:N, 1:N, final_noise_correlation_Pearson, [-1*fig_range_noise, fig_range_noise]);
colorbar;
colormap(redblue);
title('Pearson noise correlations', 'Interpreter', 'latex');
% Two-stage Pearson noise correlation estimates
subplot(2,4,4);
temp = noise_correlation_Two_Stage;
for j = 1:N
    temp(j,j)=0;
end
final_noise_correlation_Two_Stage = temp / norm(temp,2);
imagesc(1:N, 1:N, final_noise_correlation_Two_Stage, [-1*fig_range_noise, fig_range_noise]);
colorbar;
colormap(redblue);
title('Two-Stage Pearson noise correlations', 'Interpreter', 'latex');

% Performance comparison of noise correlation estimates: MSE wrt ground truth
mean_squared_error_noise_correlations_Proposed = norm(final_noise_correlation_Proposed - final_noise_correlation_ground_truth,'fro')/norm(final_noise_correlation_ground_truth,'fro');
mean_squared_error_noise_correlations_Pearson = norm(final_noise_correlation_Pearson - final_noise_correlation_ground_truth,'fro')/norm(final_noise_correlation_ground_truth,'fro');
mean_squared_error_noise_correlations_Two_Stage = norm(final_noise_correlation_Two_Stage - final_noise_correlation_ground_truth,'fro')/norm(final_noise_correlation_ground_truth,'fro');

% Performance comparison of noise correlation estimates: Leakage effect
% derived by the ratio between out-of-network and in-network power
delta_x = 0.2;
in_network_noise_correlations = abs(noise_correlation_ground_truth) > delta_x;
out_network_noise_correlations= abs(noise_correlation_ground_truth) <= delta_x;
leakage_noise_correlations_Proposed = norm(final_noise_correlation_Proposed.*out_network_noise_correlations,'fro')/norm(final_noise_correlation_Proposed.*in_network_noise_correlations,'fro');
leakage_noise_correlations_Pearson = norm(final_noise_correlation_Pearson.*out_network_noise_correlations,'fro')/norm(final_noise_correlation_Pearson.*in_network_noise_correlations,'fro');
leakage_noise_correlations_Two_Stage = norm(final_noise_correlation_Two_Stage.*out_network_noise_correlations,'fro')/norm(final_noise_correlation_Two_Stage.*in_network_noise_correlations,'fro');
%% Computing Signal correlations from all methods and plotting results

% Ground truth signal correlations
signal_covariance_ground_truth = D_true * cov(s') * D_true';
signal_correlation_ground_truth =  diag(1./sqrt(diag(signal_covariance_ground_truth)))*signal_covariance_ground_truth*diag(1./sqrt(diag(signal_covariance_ground_truth)));

% Proposed signal correlation estimates
signal_covariance_Proposed = D_hat_proposed * cov(s') * D_hat_proposed';
signal_correlation_Proposed =  diag(1./sqrt(diag(signal_covariance_Proposed)))*signal_covariance_Proposed*diag(1./sqrt(diag(signal_covariance_Proposed)));

% Pearson signal correlation estimates
signal_covariance_Pearson = cov(mean(Y, 3));
signal_correlation_Pearson =  diag(1./sqrt(diag(signal_covariance_Pearson)))*signal_covariance_Pearson*diag(1./sqrt(diag(signal_covariance_Pearson)));

% Two-Stage signal correlation estimates
signal_covariance_Two_Stage = cov(mean(spikes_smoothed_FCSS, 3)');
signal_correlation_Two_Stage =  diag(1./sqrt(diag(signal_covariance_Two_Stage)))*signal_covariance_Two_Stage*diag(1./sqrt(diag(signal_covariance_Two_Stage)));

% Plotting the results of signal correlation estimation
fig_range_signal = 0.3;
figure(4);
% Ground truth signal correlations
subplot(2,4,5);
temp = signal_correlation_ground_truth;
for j = 1:N
    temp(j,j)=0;
end
final_signal_correlation_ground_truth = temp / norm(temp,2);
imagesc(1:N, 1:N, final_signal_correlation_ground_truth,[-1*fig_range_signal, fig_range_signal]);
colorbar;
colormap(redblue);
title('Ground truth signal correlations', 'Interpreter', 'latex');
% Proposed signal correlation estimates
subplot(2,4,6);
temp = signal_correlation_Proposed;
for j = 1:N
    temp(j,j)=0;
end
final_signal_correlation_Proposed = temp / norm(temp,2);
imagesc(1:N, 1:N, final_signal_correlation_Proposed,[-1*fig_range_signal, fig_range_signal]);
colorbar;
colormap(redblue);
title('Proposed signal correlations', 'Interpreter', 'latex');
% Pearson signal correlation estimates
subplot(2,4,7);
temp = signal_correlation_Pearson;
for j = 1:N
    temp(j,j)=0;
end
final_signal_correlation_Pearson = temp / norm(temp,2);
imagesc(1:N, 1:N, final_signal_correlation_Pearson,[-1*fig_range_signal, fig_range_signal]);
colorbar;
colormap(redblue);
title('Pearson signal correlations', 'Interpreter', 'latex');
subplot(2,4,8);
% Two-Stage Pearson signal correlation estimates
temp = signal_correlation_Two_Stage;
for j = 1:N
    temp(j,j)=0;
end
final_signal_correlation_Two_Stage = temp / norm(temp,2);
imagesc(1:N, 1:N, final_signal_correlation_Two_Stage,[-1*fig_range_signal, fig_range_signal]);
colorbar;
colormap(redblue);
title('Two-Stage Pearson signal correlations', 'Interpreter', 'latex');

% Performance comparison of signal correlation estimates: MSE wrt ground truth
mean_squared_error_signal_correlations_Proposed = norm(final_signal_correlation_Proposed - final_signal_correlation_ground_truth,'fro')/norm(final_signal_correlation_ground_truth,'fro');
mean_squared_error_signal_correlations_Pearson = norm(final_signal_correlation_Pearson - final_signal_correlation_ground_truth,'fro')/norm(final_signal_correlation_ground_truth,'fro');
mean_squared_error_signal_correlations_Two_Stage = norm(final_signal_correlation_Two_Stage - final_signal_correlation_ground_truth,'fro')/norm(final_signal_correlation_ground_truth,'fro');

% Performance comparison of signal correlation estimates: Leakage effect
% derived by the ratio between out-of-network and in-network power
delta_s = 0.2;
in_network_signal_correlations = abs(signal_correlation_ground_truth) > delta_s;
out_network_signal_correlations= abs(signal_correlation_ground_truth) <= delta_s;
leakage_signal_correlations_Proposed = norm(final_signal_correlation_Proposed.*out_network_signal_correlations,'fro')/norm(final_signal_correlation_Proposed.*in_network_signal_correlations,'fro');
leakage_signal_correlations_Pearson = norm(final_signal_correlation_Pearson.*out_network_signal_correlations,'fro')/norm(final_signal_correlation_Pearson.*in_network_signal_correlations,'fro');
leakage_signal_correlations_Two_Stage = norm(final_signal_correlation_Two_Stage.*out_network_signal_correlations,'fro')/norm(final_signal_correlation_Two_Stage.*in_network_signal_correlations,'fro');

%% Plotting sample time domain estimates

z_k_K = z_k_k_psi(:,:,:,selected_setting_final); % estimated calcium concentrations
n_hat = n_hat_psi(:,:,:,selected_setting_final); % estimated spikes
m_x = m_x_psi(:,:,:,selected_setting_final); % estimated latent brain state
K1 = 2000; % range of time frames considered for the plot

selected_neuron = 1; selected_trial = 1; t1 = 1:K1;
figure(5);
subplot(3,1,1);
plot(t1, Y(t1,selected_neuron,selected_trial), 'k');
axis([1, K1, -0.05, 0.35]);
title('y_{t,1}^{(1)}');
subplot(3,1,2);
plot(t1, z_k_K(selected_neuron,t1,selected_trial), 'b', t1, n_hat(selected_neuron,t1,selected_trial), 'r');
axis([1, K1, -0.5, 3]);
title('Proposed time domain estimates', 'Interpreter', 'latex');
legend('z_{t,1}^{(1)}', 'n_{t,1}^{(1)}');
subplot(3,1,3);
plot(t1, m_x(selected_neuron,t1,selected_trial));
title('Latent brain state');
