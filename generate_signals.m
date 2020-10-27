%% This function generates the flouresence traces used as observations in the simulation study (Section 5.1).

% Outputs:
    % Y:                    the ensemble of flouresence observations (time_frames * neurons * trials)
    % alpha:                the state transition parameter (scalar)
    % mu_x:                 true mean of the latent process (1 * neurons)
    % Sigma_w:              the noise covariance of the observations (neurons * neurons)
    % true_covariance:      ground truth noise covariance of the latent noise process (neurons * neurons)
    % oracle_covariance:    the oracle estimate of the latent noise process (neurons * neurons)
    % avg_spiking_rate:     average spiking rate of the neurons under consideration (neurons * 1)
    % A:                    the scale matrix of the observations (neurons * neurons)
    % s:                    stimulus (lags * time_frames)
    % D:                    the true weight matrix of the stimulus (neurons * lags)
    % spikes:               ground truth spikes (time_frames * neurons * trials)
    
function [Y, alpha, mu_x, Sigma_w, true_covariance, avg_spiking_rate, A, s, D, spikes] = generate_signals()
%% Initilizing parameters 

K = 5000; % total number of time frames
N = 8; % number of neurons considered for the analysis
L = 20; % number of trials per neuron
alpha = 0.98; % state transition parameter
mu_x = -4.5 * ones(1,N); % mean of the latent process, set to a negative value to generate spikes at a low spiking rate
psi_x = 650*[6,0,0,0,4,0,4,0;0,6,0,-4,0,0,0,0;0,0,6,0,0,-4,0,0;0,-4,0,6,0,0,0,0;4,0,0,0,6,0,4,0;0,0,-4,0,0,6,0,0;4,0,0,0,4,0,6,3;0,0,0,0,0,0,3,6]; % true inverse Wishart scale matrix
m_x = N + 2000; % true inverse Wishart normalizing factor
A = 0.1*eye(N); % scale matrix of the observations
Sigma_w = 2*10^(-4) *eye(N); % Observation noise covariance

% We generate the true noise covariance matrix from an inverse Wishart distribution
true_covariance = iwishrnd(psi_x, m_x);

% the true weight vector
M = 2; % number of time lags of the stimulus, taken into consideration
D = zeros(N,M); % the true weight vector
D(:,1) =  [0.5, -0.3, -0.3, -0.3, 0.3, -0.3, -0.3, -0.3]'; 
D(:,2) = [0.2, 0.3,  0.3, -0.2, -0.3, -0.3, -0.3, 0.3]';

%% Generating the combined process that govern spiking activity

% We generate the stimulus by a 6th order AR process
f = 10; % tuned frequency
fs = 100; % sampling frequency
sigma_input = 3 * 10^-5; % input noise variance
input = randn(size( 0:1:2*K));  % input noise
input = input - mean(input);
b = conv(conv(conv([1 -0.99*exp(1j*2*(pi)*f/fs)],[1 -0.99*exp(-1j*2*(pi)*f/fs)]),conv([1 -0.99*exp(1j*2*(pi)*f/fs)],[1 -0.99*exp(-1j*2*(pi)*f/fs)])),conv([1 -0.99*exp(1j*2*(pi)*f/fs)],[1 -0.99*exp(-1j*2*(pi)*f/fs)])); % setting poles of the AR process
s_temp = filter(1,b,sigma_input*input);

s_initial = s_temp(end-K+1-M:end) - 1; % this is the external stimulus that govern spiking activity
s = zeros(M, K);
for k = 1:K
    s(:,k) = flip(s_initial(k:k+M-1));
end

%% Generating spikes according to the logistic model

spikes = zeros(K,N,L); % Spike train generation
for l = 1:L
    X_latent = mvnrnd(mu_x,true_covariance,K); % the latent Gaussian noise process
    X_combined = X_latent +  s'*D'; % the overall combined governing process (latent process + contribution from stimulus)
    lambda = 1 ./ (1 + exp(-1*X_combined)); % CIF (Conditional Intensity Function)
    spikes(:,:,l) = rand(size(X_combined)) < lambda;
end

% Average spiking rate of each neuron
avg_spiking_rate = zeros(N,1);
for i = 1:N
avg_spiking_rate(i) = length(find(spikes(:,i,:) > 0)) / (K*L);
end

%% Generating the flouresence observations

% Generating the calcium concentrations
Z = zeros(size(spikes));
Z(1, :, :) = spikes(1, :, :);
for k = 2:K
    Z(k, :, :) = alpha * Z(k-1, :, :) +  spikes(k, :, :);
end

% Generating the noisy flouresence observations (Y)
Y = Z;
for l = 1:L
    for k = 1:K
        Z(k,:,l) = A * squeeze(squeeze(Z(k,:,l)))';
    end
    noise_process_w = mvnrnd(zeros(1,N),Sigma_w,K);
    Y(:,:,l) = Z(:,:,l) + noise_process_w;
end

%% Ground truth correlations

%Noise correlations
noise_correlations_ground_truth = diag(1./sqrt(diag(true_covariance)))*true_covariance*diag(1./sqrt(diag(true_covariance)));

%Signal correlations
signal_covariance_ground_truth = D * cov(s') * D';
signal_correlation_ground_truth =  diag(1./sqrt(diag(signal_covariance_ground_truth)))*signal_covariance_ground_truth*diag(1./sqrt(diag(signal_covariance_ground_truth)));

%% Plotting the ground truth
figure(2);

% Noise correlations
subplot(2,2,1);
temp = noise_correlations_ground_truth;
for j = 1:N
    temp(j,j)=0;
end
final_noise_ground_truth = temp / norm(temp,2);
imagesc(1:N, 1:N, final_noise_ground_truth, [-0.3, 0.3]);
colorbar;
colormap(redblue);
title('Ground truth noise correlation', 'Interpreter', 'latex');
% Signal correlations
subplot(2,2,3);
temp = signal_correlation_ground_truth;
for j = 1:N
    temp(j,j)=0;
end
final_signal_ground_truth = temp / norm(temp,2);
imagesc(1:N, 1:N, final_signal_ground_truth, [-0.3, 0.3]);
colorbar;
colormap(redblue);
title('Ground truth signal correlation', 'Interpreter', 'latex');