%% This function evaluates the performance of the estimates at each hyper parameter setting (see Suppl. Section S5 for details)

% Inputs: 
    % Y:                the ensemble of flouresence observations (time_frames * neurons * trials)
    % cov_estimated:    the current noise covariance estimate (neurons * neurons)
    % Sigma_w:          the noise covariance of the observations (neurons * neurons)
    % alpha:            the state transition parameter (scalar)
    % A:                the scale matrix of the observations (neurons * neurons)
    % mu_x:             the true mean of the latent process (1 * neurons)
    % D_hat:            the current weight matrix estimate (neurons * lags)
    % s:                the external stimulus with time lags (lags * time_frames)

% Outputs:
    % performance:      the disparity between the true observation covariance and simulated observation covariance

function performance = performance_evaluation(Y, cov_estimated, Sigma_w, alpha, A, mu_x, D_hat, s)
%% Intializing the variables

K = size(Y, 1);
N = size(Y, 2); 
L = size(Y, 3);

% The empirical covariance of the true flouresence observations
covariance_Y_true = zeros(N);
for l = 1:L
    covariance_Y_true = covariance_Y_true + cov(squeeze(Y(:,:,l)));
end
covariance_Y_true = covariance_Y_true/L;

%% Simulating new observations using the model and current parameter estimates

% Simulated spike train
spikes_simulated = zeros(K,N,L);
for l = 1:L
    % Simulated combined latent process
    X_simulated = mvnrnd(mu_x,cov_estimated,K);
    X_combined_simulated = X_simulated + s'*D_hat';
    lambda = 1 ./ (1 + exp(-1*X_combined_simulated)); % CIF
    spikes_simulated(:,:,l) = rand(size(X_combined_simulated)) < lambda;
end

% Simulated calcium concentrations
Z_simulated = zeros(size(spikes_simulated));
Z_simulated(1, :, :) = spikes_simulated(1, :, :);
for k = 2:K
    Z_simulated(k, :, :) = alpha * Z_simulated(k-1, :, :) +  spikes_simulated(k, :, :);
end

% Simulated flouresence observations
Y_simulated = Z_simulated;
for l = 1:L
    for k = 1:K
        Z_simulated(k,:,l) = A * squeeze(squeeze(Z_simulated(k,:,l)))';
    end
    noise_process_w = mvnrnd(zeros(1,N),Sigma_w,K);
    Y_simulated(:,:,l) = Z_simulated(:,:,l) + noise_process_w;
end

% The empirical covariance of the simulated observations
covariance_Y_simulated = zeros(N);
for l = 1:L
    covariance_Y_simulated = covariance_Y_simulated + cov(squeeze(Y_simulated(:,:,l)));
end
covariance_Y_simulated = covariance_Y_simulated/L;

%% Evaluating the disparity between the true observation covariance and the simulated observation covariance based on Frobenius norm

performance = norm(covariance_Y_simulated - covariance_Y_true,'fro');

