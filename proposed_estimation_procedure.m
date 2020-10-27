%% This function implements the proposed iterative algorithm for noise covariance (Sigma_x) and weight matrix (D_hat) estimation

% Inputs:
    % Y:                the ensemble of flouresence observations (time_frames * neurons * trials)
    % Sigma_w:          the noise covariance of the observations (neurons * neurons)
    % mu_x:             the true mean of the latent process (1 * neurons)
    % gamma_x:          the inverse Wishart hyper-parameter of the scale factor (scalar)
    % psi_x:            the inverse Wishart hyper-parameter of the scale matrix (neurons * neurons)
    % alpha:            the state transition parameter (scalar)
    % A:                the scale matrix of the observations (neurons * neurons)
    % s:                the external stimulus with time lags (neurons * lags)
    % D_hat:            the initialization of the weight matrix (lags * time_frames)
    % error_threshold:  the threshold of accepted error
    
% Outputs:
    % Sigma_x:          the current noise covariance matrix estimate (neurons * neurons)
    % z_k_K:            the current calcium concentration estimates (neurons * time_frames * trials)
    % n_hat:            the current spike estimates (neurons * time_frames * trials)
    % D_hat:            the current weight matrix estimate (lags * time_frames)
    % m_x:              the current estimated mean of the latent brain state (neurons * time_frames * trials)
    
function [Sigma_x, z_k_K, n_hat, D_hat, m_x] = proposed_estimation_procedure(Y, Sigma_w, mu_x, gamma_x, psi_x, alpha, A, s, D_hat,error_threshold) 
%% Initializing the variables and parameters

% Setting auxiliary hyper-parameters
maximum_iterations = 250;
beta = 8; % hyper-parameter >= 1
epsilon = 10^-6; % epsilon: IRLS perturbation 
% error_threshold = 0.02; % threshold of accepted error

% Initializing variables
K = size(Y, 1);  % total number of time frames
N = size(Y, 2); % number of neurons considered for the analysis
L = size(Y, 3); % number of trials per neuron
M = size(s,1); % time lags of the stimulus

% FIS parameter initialization
z_k_k = zeros(N, K, L); % k|k
z_k_K = zeros(N, K, L); % k|K
z_k_k_1 = zeros(N, K, L);   % k|k-1
P_k_k = zeros(N, N, K, L);   % k|k
P_k_k_1 = P_k_k;   % k|k-1
P_k_K = P_k_k;  % k|K 
B = zeros(N, N, K, L);

% VI parameter initialization
n_hat = (10^-3)*ones(N, K, L);
m_x = mu_x(1)*ones(N, K, L);
Q_x = zeros(N, N, K, L);
P_x = 10^-5 * eye(N);
c_PG = 10^-3*ones(N, K, L);
for j = 1:N
    c_PG(j,:,:) =  sqrt((mu_x(j))^2 + P_x(j,j)) * c_PG(j,:,:);
end
omega = tanh(c_PG / 2) ./ (2*c_PG);
nu = -beta*m_x;
Sigma_nu = P_k_k;
for l = 1:L
    for k = 1:K
        Sigma_nu(:,:,k,l) = diag(epsilon*ones(N,1) ./ nu(:,k,l));
    end
end
Sigma_x = zeros(N,N);

current_error = inf;
iteration = 1;

%% The overall iterative procedure

while (iteration <= maximum_iterations) && (current_error > error_threshold)
    
    % Estimating the calcium concentrations using Fixed Interval Smoothing
    for l = 1:L        
        % Forward filtering
        for k = 1:K
            if k == 1
                if iteration == 1
                    z_k_k_1(:, k,l) = zeros(N,1);
                    P_k_k_1(:, :, k,l) = Sigma_nu(:,:,k,l);
                else
                    z_k_k_1(:, k,l) = z_k_K(:,k,l);
                    P_k_k_1(:, :, k,l) = P_k_K(:,:,k,l);
                end
            else
               z_k_k_1(:, k, l) = alpha * z_k_k(:, k-1, l);
               P_k_k_1(:, :, k, l) = (alpha^2) * P_k_k(:, :, k-1, l) + Sigma_nu(:,:,k,l);
            end
            B(:, :, k, l) = squeeze(squeeze(P_k_k_1(:, :, k, l)))*A' / (A*squeeze(squeeze(P_k_k_1(:, :, k, l)))*A' + Sigma_w);
            z_k_k(:, k, l) = z_k_k_1(:, k, l) + B(:, :, k, l) * ((Y(k,:,l))' - A*z_k_k_1(:, k,l));
            P_k_k(:, :, k, l) = (eye(N) - B(:, :, k, l)*A) * P_k_k_1(:, :, k, l);        
        end
        % Backward smoothing
        z_k_K(:, K,l) = z_k_k(:, K,l); P_k_K(:,:,K,l) = P_k_k(:,:,K,l);
        for k = K-1:-1:1
            z_k_K(:, k,l) = z_k_k(:, k, l) + alpha * P_k_k(:, :, k, l) / P_k_k_1(:, :, k+1,l) * (z_k_K(:, k+1,l) - z_k_k_1(:, k+1, l));
        end   
    end
    
    %Updating variational parameters
    n_hat_prev = n_hat;
    for k = 1:K
        if k == 1
            n_hat(:,k, :) = z_k_K(:, k, :);
        else
            n_hat(:,k, :) = z_k_K(:, k, :) - alpha * z_k_K(:, k-1, :);
        end
        for l = 1:L
        Q_x(:,:,k,l) = (gamma_x * (P_x \ eye(N)) + diag(squeeze(squeeze(omega(:,k,l))), 0)) \ eye(N);
        m_x(:,k,l) = squeeze(squeeze(Q_x(:,:,k,l))) * ((gamma_x ) * (P_x \ eye(N)) * mu_x' + n_hat(:,k,l) - 1/2 - diag(squeeze(squeeze(omega(:,k,l))), 0)*D_hat*squeeze(s(:,k)));
        c_PG(:,k,l) = sqrt((squeeze(squeeze(m_x(:,k,l))) + D_hat*squeeze(s(:,k))).^2 + diag(squeeze(squeeze(Q_x(:,:,k,l)))));
        nu(:,k,l) = beta* abs(m_x(:,k,l) + D_hat*squeeze(s(:,k)));
        
        % Update IRLS covariance approximation
            if k == 1
                Sigma_nu(:,:,k,l) = diag(sqrt((squeeze(squeeze(z_k_K(:, k,l)))).^2 + epsilon^2 * ones(N,1))./squeeze(squeeze(nu(:,k,l))));
            else
                Sigma_nu(:,:,k,l) = diag(sqrt((squeeze(squeeze(z_k_K(:, k,l))) - alpha*squeeze(squeeze(z_k_K(:, k-1,l)))).^2 ...
                    + epsilon^2 * ones(N,1))./squeeze(squeeze(nu(:,k,l))));
            end
        end
    end
    
    omega = tanh(c_PG / 2) ./ (2*c_PG);
    
    % Update outputs and the convergence criterion
    P_x = psi_x;
    for l = 1:L
        for k = 1:K
            P_x = P_x + (squeeze(squeeze(m_x(:,k,l)))) * (squeeze(squeeze(m_x(:,k,l))))' + squeeze(squeeze(Q_x(:,:,k,l))) - (squeeze(squeeze(m_x(:,k,l)))) * mu_x ...
            - mu_x' * (squeeze(squeeze(m_x(:,k,l))))' + mu_x' * mu_x;
        end
    end   
    for j = 1:N
        numerator_d_j = zeros(M,1);
        denomenator_d_j = zeros(M,M);
        for k = 1:K
            for l = 1:L
                numerator_d_j = numerator_d_j + (n_hat(j,k,l) - 1/2 -  omega(j,k,l) * m_x(j,k,l))*squeeze(s(:,k));
                denomenator_d_j = denomenator_d_j + omega(j,k,l)*squeeze(s(:,k))*squeeze(s(:,k))';
            end
        end
        d_j = denomenator_d_j \ numerator_d_j;
        D_hat(j,:) = d_j';
    end
    
    Sigma_x = P_x/(gamma_x + N + 1);
    
    % iterating until the estimates of spikes converge
    current_error = sum(sum(sum(abs(n_hat - n_hat_prev))))/sum(sum(sum(abs(n_hat_prev)))) 
    iteration = iteration + 1
   
    % Plotting Proposed current results
    
    % Plotting sample time domain estimates
    selected_neuron = 1; selected_realization = 1; t = 1:K;
    figure(1);
    subplot(2,1,1);
    plot(t, Y(:,selected_neuron,selected_realization), 'k');
    axis([1, K, -0.05, 0.4]);
    title('y_{t,1}^{(1)}');
    subplot(2,1,2);
    plot(t, z_k_K(selected_neuron,:,selected_realization), 'b', t, n_hat(selected_neuron,:,selected_realization), 'r');
    axis([1, K, -0.5, 4]);
    title('Current time domain estimates', 'Interpreter', 'latex');
    legend('z_{t,1}^{(1)}', 'n_{t,1}^{(1)}');
    drawnow;
    
    % Plotting current Proposed correlation estimates
    figure(2);
    % Current noise correlation matrix
    subplot(2,2,2);
    noise_correlation_estimate = diag(1./sqrt(diag(Sigma_x))) * Sigma_x * diag(1./sqrt(diag(Sigma_x)));
    temp = noise_correlation_estimate;
    for j = 1:N
        temp(j,j)=0;
    end
    noise_correlation_proposed = temp / norm(temp,2);
    imagesc(1:N, 1:N, noise_correlation_proposed, [-0.3, 0.3]);
    colorbar;
    colormap(redblue);
    title('Current Proposed noise correlation estimate', 'Interpreter', 'latex');
    % Current signal correlation matrix
    subplot(2,2,4); 
    signal_covariance_estimate = D_hat * cov(s') * D_hat';
    signal_correlation_estimate =  diag(1./sqrt(diag(signal_covariance_estimate)))*signal_covariance_estimate*diag(1./sqrt(diag(signal_covariance_estimate)));
    temp = signal_correlation_estimate;
    for j = 1:N
        temp(j,j)=0;
    end
    signal_correlation_proposed = temp / norm(temp,2);
    imagesc(1:N, 1:N, signal_correlation_proposed,[-0.3, 0.3]);
    colorbar;
    colormap(redblue);
    title('Current Proposed signal correlation estimate', 'Interpreter', 'latex');
    drawnow;
    
end

  