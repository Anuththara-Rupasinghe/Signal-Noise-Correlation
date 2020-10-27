%% This function generates the deconvolved spikes from flouresence observations, via the FCSS method.

% Inputs:
     % Y:            the ensemble of flouresence observations (time_frames * neurons * trials)
     % scale_matrix: the scale matrix of the observations (neurons * neurons)

% Output:
    % estimated_spikes_FCSS: the deconvolved spikes via the FCSS method (neurons * time_frames * trials)

function estimated_spikes_FCSS = spike_deconvolution_FCSS(Y, scale_matrix)

    N = size(Y, 2); % number of neurons
    K = size(Y, 1); % number of time frames
    L = size(Y, 3); % number of independent realization

    estimated_spikes_FCSS = zeros(N,K,L);

    for l = 1:L   
        sys.y = squeeze(Y(:,:,l))';
        sys.lambda = 0.5; %Increase for smoother estimates/decrease for noisier
        sys.maxNumIters = 10;      % Increase for better converge (slower) and vice versa
        % sys.EMFlag = false;        % can turn EM off
        % sys.baseline = zeros(p,1); % can input baseline manually
        sys.C = scale_matrix;
        sys_smoothed = FCSS_IRLS(sys);
        estimated_spikes_FCSS(:,:,l) = sys_smoothed.spikes > 0;
    end

end
%% 

function baseline = FCSS_calc_base(sys)
% calculates baseline of the traces
[n,T] = size(sys.y);
baseline = min(sys.y,[],2);
sigma = sqrt(diag(sys.R));
p = length(sigma);
if isequal(n,p)
    y = sys.y-repmat(baseline,1,T);
    for i = 1:n
        a = y(i,:);
        a = a(a<3*sigma(i));
    baseline(i) = mean(a);
    end
else
    disp('Warning: number of measurements (n) is less than the ambient dimension (p)')
    disp('Baseline set to minimum of y !')
end

end

%% 

function [X Y] = FCSS_calc_Calcium_IP(coeffs)
num_coeffs = length(coeffs);
if num_coeffs < 5; error('There should be 5 coefficients for Calcium parameter estimation'); end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Need to be careful about the order
cx2 = coeffs(1); if cx2 <= 0; error('Non-Convex Problem'); end
cy2 = coeffs(2); if cy2 <= 0; error('Non-Convex Problem'); end
cxy = coeffs(3);
cx  = coeffs(4);
cy  = coeffs(5);

%%%%%%%%%%%%%%%%%%%%%%%%%
A = [2*cx2 cxy; cxy 2*cy2]\[-cx; -cy];
X = A(1); Y = A(2);
%%%%%%%%%%%%%%%% If not satisfying constraints go and check the edges of the region
if check_constraints1(X,Y)
    
    r(1) = abs(X+sqrt(X^2+4*Y))/2;
    r(2) = abs(X-sqrt(X^2+4*Y))/2;
    X = X*.99/max(r);
    Y = Y*.99^2/max(r)^2;
end

end

function c = check_constraints1(X,Y)
c = abs(X+sqrt(X^2+4*Y))/2 > 0.999 || abs(X-sqrt(X^2+4*Y))/2 > 0.99 || X^2+4*Y < 0;
end

%% 

function [Q, W] = FCSS_calc_Q(X, A, epsilon,C )

[p,T] = size(X);
if nargin<4
    n = p;
else
    n = size(C,1);
end

if ~isequal(n,p)
    X = pinv(C)*X;
end
Xdiff = zeros(size(X));

for i = 1:p
    % filtering dimension does not matter
% Xdiff(i,:) = sqrt( (filter([1 -full(A(i,:))],1,X(i,:),[],2)).^2+epsilon^2);
Xdiff(i,:) = sqrt(filter([1 -full(A(i,:))],1,X(i,:)).^2+epsilon^2);
end
S = sparse(repmat(1:p,1,T),1:p*T,Xdiff(1:end),p,p*T );
Q = reshape(full(S),p,p,T);

XdiffInv = 1./Xdiff;

WS = sparse(repmat(1:p,1,T),1:p*T,XdiffInv(1:end),p,p*T );
W = reshape(full(WS),p,p,T);

% 
% WS0 = vec(XdiffInv)';
% WS1 = repmat(WS0,p,1);
% W = reshape(WS1,p,p,T);

end

%% 

function [X Y] = FCSS_calc_spindle_IP(coeffs,fs,min_freq,max_freq)
num_coeffs = length(coeffs);
if num_coeffs < 5; error('There should be 5 coefficients for spindle parameter estimation'); end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 3; min_freq = 12; max_freq = 14; end
if nargin < 2; fs = 200; end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Need to be careful about the order
cx2 = coeffs(1); if cx2 <= 0; error('Non-Convex Problem'); end
cy2 = coeffs(2); if cy2 <= 0; error('Non-Convex Problem'); end
cxy = coeffs(3);
cx  = coeffs(4);
cy  = coeffs(5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ly = -.99^2;
uy = -.95^2;
% lx2 = -4 * ly * cos(2*pi*min_freq/fs);
% ux2 = -4 * uy * cos(2*pi*max_freq/fs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Y = - a^2; X = 2a cos(2pi f/fs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization Problem : (Non-Convex)
% minimize       cx2   X^2 +  cy2  Y^2  + cxy XY + cx X + cy Y
% subject to    lambda3:  -4Y cos(2*pi max_freq/fs)^2    <=  X^2 <= -4Y cos(2*pi  min_freq/fs)^2 : lambda4                  
%               lambda1:                            ly  <=  Y <=   uy                        : lambda2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 cy2 Y + cxy X + cy + lambda2 -  lambda1 + 4 lambda4 cos(2*pi  min_freq/fs)^2 - 4 lambda3 cos(2*pi  max_freq/fs)^2 = 0
% 2 cx2 X + cxy Y + cx + 2 lambda4 X - 2 lambda3 X = 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complementary Slackness
% (2 cy2 Y + cxy X + cy)(Y-ub)(Y-lb)(X^2 + 4Y cos(2*pi  min_freq/fs) )^2( X^2 + 4Y cos(2*pi  max_freq/fs) )^2 = 0
% (2 cx2 X + cxy Y + cx)(X^2 + 4Y cos(2*pi  min_freq/fs) )^2( X^2 + 4Y cos(2*pi  max_freq/fs) )^2 = 0


%%%%%%%%%%%%%%%%%%%%%%%%%
A = [2*cx2 cxy; cxy 2*cy2]\[-cx; -cy];
X = A(1); Y = A(2);
%%%%%%%%%%%%%%%% If not satisfying constraints go and check the edges of the region
if check_constraints(X,Y,uy,ly,min_freq,max_freq,fs)
    
    % case 1
    Y(1) = ly;
    X(1) = (-cx-cxy*Y(1))/(2*cx2);
    a(1) = f(X(1),Y(1),uy,ly,min_freq,max_freq,fs,coeffs);
    
    Y(2) = ly;
    X(2) = sqrt(-4*Y(2))*cos(2*pi*  min_freq/fs);
    a(2) = f(X(2),Y(2),uy,ly,min_freq,max_freq,fs,coeffs);
    
    Y(3) = ly;
    X(3) = sqrt(-4*Y(3))*cos(2*pi*  max_freq/fs);
    a(3) = f(X(3),Y(3),uy,ly,min_freq,max_freq,fs,coeffs);
    
    % case 2
    Y(4) = uy;
    X(4) = (-cx-cxy*Y(4))/(2*cx2);
    a(4) = f(X(4),Y(4),uy,ly,min_freq,max_freq,fs,coeffs);
     
    Y(5) = uy;
    X(5) = sqrt(-4*Y(5))*cos(2*pi*  min_freq/fs);
    a(5) = f(X(5),Y(5),uy,ly,min_freq,max_freq,fs,coeffs);
    
    Y(6) = uy;
    X(6) = sqrt(-4*Y(6))*cos(2*pi*  max_freq/fs);
    a(6) = f(X(6),Y(6),uy,ly,min_freq,max_freq,fs,coeffs);
    
    % case 3
    syms z;
    S = solve(2*cx2*z - 3/4*cxy*z^2/cos(2*pi*min_freq/fs)^2 + cy2*z^3/4/cos(2*pi*min_freq/fs)^4+cx-cy*z/2/cos(2*pi*min_freq/fs)^2 ...
        ,z>0,z<2*(cos(2*pi*min_freq/fs)));
    
    for i=1:length(S)
    X(end+1) = S(i);    
    Y(end+1) = -X(end).^2/4/cos(2*pi*min_freq/fs)^2;
    a(end+1) = f(X(end),Y(end),uy,ly,min_freq,max_freq,fs,coeffs);
    end
    
    S = solve(2*cx2*z - 3/4*cxy*z^2/cos(2*pi*max_freq/fs)^2 + cy2*z^3/4/cos(2*pi*max_freq/fs)^4+cx-cy*z/2/cos(2*pi*max_freq/fs)^2 ...
        ,z>0,z<2*(cos(2*pi*min_freq/fs)));
    
        for i=1:length(S)
    X(end+1) = S(i);    
    Y(end+1) = -X(end).^2/4/cos(2*pi*max_freq/fs)^2;
    a(end+1) = f(X(end),Y(end),uy,ly,min_freq,max_freq,fs,coeffs);
        end

        [m, I] = min(a);
        X = X(I);
        Y = Y(I);
%         check_constraints(X,Y,uy,ly,min_freq,max_freq,fs)
%         a = sqrt(-Y)
%         f = fs*acos(X/2/a)/2/pi
end

end

function c = check_constraints(X,Y,uy,ly,min_freq,max_freq,fs)
c = (Y > uy) || (Y < ly) || (X^2 > -4*Y* cos(2*pi*  min_freq/fs)^2) || (X^2 < -4*Y* cos(2*pi*  max_freq/fs)^2);
end

function a = f(X,Y,uy,ly,min_freq,max_freq,fs,coeffs)

if ~check_constraints(X,Y,uy,ly,min_freq,max_freq,fs); a = coeffs(1)*X^2+coeffs(2)*Y^2+coeffs(3)*X*Y+coeffs(4)*X+coeffs(5)*Y;
else a = Inf;   end
    

end

%% 

function A_block = FCSS_calc_transition_matrix(A,p,Order)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$%
    % Create a block diagonal matrix with i-th block having parameters of
    % the ith ROI which is
    % ai or  [ai(1) ai(2)]   for AR(1) and AR(2) respectively
    %        [1      0   ]
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch Order
    case 1
        if isequal(size(A,1),size(A,2))
        A_block = A;
        else
            A_block = diag(A);
        end
        
    case 2
    Elements = A';
    % diagonal elements
    D = sparse(1:2*p,1:2*p,upsample(Elements(1,:),2),2*p,2*p);
    % upper diagonal elements
    r = upsample(Elements(2,:),2);
    U = sparse(1:2*p-1,2:2*p,r(1:end-1),2*p,2*p);
    % lower diagonal elements
    c = upsample(ones(1,p),2);
    L = sparse(2:2*p,1:2*p-1,c(1:end-1),2*p,2*p);
    % tridiagonal 
    A_block = D+L+U;

end

end

%% 

function R = FCSS_calc_var( y )

[p,T] = size(y);
range_ff =[ 0.25,0.5];
        [psd_Y,ff]= pwelch(y',round(T/8),[],1000,1);
        psd_Y = psd_Y';
        ind = double(ff>range_ff(1));
        ind(ff>range_ff(2))= 0;
        ind(ind == 0) = NaN;
        ind = repmat(ind',p,1);
        R = diag(exp(nanmean(log(psd_Y.*ind/2),2)));

end

%% 

function A = FCSS_Expectation_Maximization(sys_smoothed,W)

% Expectation Maximization Algorithm for estimating the state-transition
% matrix A from the output of a Kalman Smoother
%%%%%%%%%%%%%%%%%%% Model %%%%%%%%%%%%%%%%%%%%%
% x_t = A x_{t-1} + w_t
% y_t = C x_t + z_t
%%%%%%%%%%% Optimization Problem  %%%%%%%%%%%%%
% minimize  Sum E(||x_t - A x_{t-1} ||_1 
%    A      t>1                         
% The problem is solved by IRLS method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Order = sys_smoothed.Order;
X = sys_smoothed.X_smoothed;
Sigma = sys_smoothed.Sigma_smoothed;
Sigma_st = sys_smoothed.Sigma_st;
[p,T] = size(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sigma(Sigma<0) = eps;
% Sigma_st(Sigma_st<0) = eps;
% Sigma_st(:,:,end) = Sigma_st(:,:,end) + (Sigma_st(:,:,end-1)>0)*eps;
switch Order
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 1
        mode = 'EM';
        switch mode
            case 'EM'
        Weights = reshape(W(W>0),p,T);        
        X0 = mult_shift(X,X,0); % X_t X_t
        
        X1 = mult_shift(X,X,1); % X_t X_{t-1}

        S = reshape(Sigma,p^2,T);
        
        S0 = S(1:p+1:end,:); % Sigmat-1t-1  
       
        S = reshape(Sigma_st,p^2,T);
        
        S1 = S(1:p+1:end,:); % Sigmatt-1

        num = sum(mult_shift(Weights,X0+S0,1),2);
        denom = sum(mult_shift(Weights,X1+S1,0),2);
        
        A = num./denom;
    
        A = min(A,.999);
    
            case 'yule'
            A = aryule(X(1:2:end,:)',Order);
        end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 2
    if ~isfield(sys_smoothed,'mode'); mode = 'EM';
    else mode = sys_smoothed.mode;
    end
    
    switch mode
     
        case 'EM'
            %%%%%% Calculate the terms in the equation for EM
        Weights = reshape(W(W>0),p/2,T);
        X = X(1:2:end,:);
        X0 = mult_shift(X,X,0); % X_t X_t
        X1 = mult_shift(X,X,1); % X_t X_{t-1}
        X2 = mult_shift(X,X,2); % X_t X_{t-2}
        
        
        S = reshape(Sigma,p^2,T);
        
        S0 = S(1:2*p+2:end,:); % Sigmat-1t-1  
        
        S1 = S(2:2*p+2:end,:); % Sigmatt-1
        
        S = reshape(Sigma_st,p^2,T);
        
        S2 = S(p+2:2*p+2:end,:); % Sigmatt-2
        
        
        coeffs = zeros(p/2,5);
        coeffs(:,1) = sum(mult_shift(Weights,X0+S0,1),2);
        coeffs(:,2) = sum(mult_shift(Weights,X0+S0,2),2);
        coeffs(:,3) = 2*sum(mult_shift(Weights,X1+S1,1),2);
        coeffs(:,4) = -2*sum(mult_shift(Weights,X1+S1,0),2);
        coeffs(:,5) = -2*sum(mult_shift(Weights,X2+S2,0),2);

        for i=1:p/2
        [A(i,1),A(i,2)] = calc_Calcium_IP(coeffs(i,:));
        end
        
        case 'yule'
            Weights = reshape(W(W>0),p/2,T);
            A = aryule((Weights.*X(1:2:end,:))',Order);

        case 'burg'
            A = arburg((Weights.*X(1:2:end,:))',Order);

            
            
        case 'spindle'
            fs = sys_smoothed.fs;
            min_freq = 12;
            max_freq = 14;
            %%%%%% Calculate the terms in the equation for EM
            %%%%%% x_{t-1} * x_{t-1}
        
        Weights = reshape(W(W>0),p/2,T);
        X = X(1:2:end,:);
        X0 = mult_shift(X,X,0); % X_t X_t
        X1 = mult_shift(X,X,1); % X_t X_{t-1}
        X2 = mult_shift(X,X,2); % X_t X_{t-2}
        
        
        S = reshape(Sigma,p^2,T);
        
        S0 = S(1:2*p+2:end,:); % Sigmat-1t-1  
        
        S1 = S(2:2*p+2:end,:); % Sigmatt-1
        
        S = reshape(Sigma_st,p^2,T);
        
        S2 = S(p+2:2*p+2:end,:); % Sigmatt-2
        
        coeffs = zeros(p/2,5);
        coeffs(:,1) = sum(mult_shift(Weights,X0+S0,1),2);
        coeffs(:,2) = sum(mult_shift(Weights,X0+S0,2),2);
        coeffs(:,3) = 2*sum(mult_shift(Weights,X1+S1,1),2);
        coeffs(:,4) = -2*sum(mult_shift(Weights,X1+S1,0),2);
        coeffs(:,5) = -2*sum(mult_shift(Weights,X2+S2,0),2);
        
        A = zeros(p/2,2);
        for i=1:p/2
        [A(i,1),A(i,2)] = calc_spindle_IP(coeffs(i,:),fs,min_freq,max_freq);
        end
        
    end
    
end


end

function Z = mult_shift(W,X,s)
M = zeros(size(W));
M(:,1:end-s) = W(:,s+1:end);
Z = M.*X;
end

%% 

function [spikes,decon,decon2]  = FCSS_Find_Spikes( sys, confidence_level )

% Inputs: structure sys: Output of the IRLS method                                     %
% confidence_level: Desired confidence level, a scalar in (0,1), default: 0.9          %
%%%%%%%%%%%%%%% Set default confidence level %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<2; confidence_level = 0.9; end                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate of how big the error will get relatively due to the differencing            %      
% operator, always set it to 1.                                                        %
alpha = 1;                                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[p, T] = size(sys.X_smoothed);                                                         %
n = size(sys.y,1);                                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deconvolved Signal                                                                   %
decon = zeros(p,T);                                                                    %
for i = 1:p                                                                            %
    decon(i,:) = filter([1 -sys.A(i,:)],1,sys.X_smoothed(i,:),[],2);                   %
    decon2(i,:) = filter([1 -1.999 1],1,sys.X_smoothed(i,:),[],2);                     %
end                                                                                    %
decon(:,1)   = 0; decon(decon<0) = 0;                                                    %
decon2(:,1:2)= 0; decon2(decon2<0) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch n                                                                               %
    case p                                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The k-th element of confidence is the confidence bound for the k-th Neuron           %
% See [Van De Geer 2014] for the derivation                                            %
confidence = 1/2*norminv(1-(1-confidence_level)/2)*sqrt(diag(sys.R));                  %
confidence = repmat(confidence,1,T);                                                   %
spikes = decon;                                                                        %
spikes(spikes<confidence) = 0;                                                         %
spikes(:,1:end-2) = spikes(:,3:end); spikes(:,end) = 0; 
spikes(find_dec(sys.X_smoothed)) = 0;
% spikes(find_dec(decon)) = 0;
decon2(spikes == 0) = 0;
for i=1:p
spikes(i,:) = FCSS_merge_spikes(spikes(i,:));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    otherwise                                                                          %
        disp('Confidence bounds for Compressive Regime Will be added soon')            %
% Augmented_meas_mtx = toeplitz(ones(T,1),[1;zeros(T-1,1)]);                           
% Augmented_Cov = Augmented_meas_mtx'*Augmented_meas_mtx/T;
% Bias =  (Augmented_Cov\(Augmented_meas_mtx*(sys.y-X_smoothed)')/T)';                 %
% X_smoothed_unbiased = sys.deconv + Bias;                                             %
% X_smoothed_unbiased = X_smoothed + Bias*Augmented_meas_mtx';                         %
% Compensate for the bias caused by regularization                                     %
end                                                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end


function [X_dec] = find_dec(X)
%%%% Locations where calcium has decreased
X_dec = filter([1 -1],1,X,[],2)<0;
%%%% Locations where calcium has increased
% X_inc = ~X_dec;
% X_dec = X_dec;
% X_inc = X_inc.*X;
end

%% 

function  sys_smoothed = FCSS_IRLS(sys)

% Iterative Reweighted Least Squares Algorithm for Autoregressive signal deconvolution Model % 
%           x(t) = A(t) x(t-1) + w(t),  w(t) ~ Laplace i.i.d. (lambda)                       %
%           y(t) = C(t) x(t)   + v(t),  v(t) ~ N(0,R(t))                                     %
%                                                                                            %
% Input struct "sys" should include the following fields                                     %
%           sys.y : Observed Calcium Signal, each row is the activity of a single neuron     %
%                                                                                            %
%           sys.lambda: regularization parameter, could choose via cross-validation          %
%                                                                                            %
% Optional: sys.C : The measurement MATRIX C, deafult: Identity (denoising)                  %
%           Should be given for Compressive Calcium Imaging                                  %
%                                                                                            %
%           sys.R : Initial estimate of the observation noise covariance MATRIX              %
%           default: R = sn from FOOPSI (Pnevmatikakis et al.(2016)                          %
%                                                                                            %
%           sys.A : A for a single neuron (scalar or 2 dimensional vector [a b])             %
%           where x(t) = a x(t-1) + b x(t-1) + w(t)                                          %
%           Warning: This will also turn off the EM steps                                    %
%                                                                                            %
%           sys.Order: Autoregressive model order, default: 1                                %
%           default: initialize with aryule(sys.y,sys.Order)                                 %
%                                                                                            %
%           sys.maxNumIters chooses the number of IRLS iterations, default: 5                %
%                                                                                            %
%           sys.resetRFlag if set to true, the estimate of the observation noise covariance  %
%           matrix gets updated, default: false                                              %
%                                                                                            %
%           sys.EMFlag if set to false, EM updates of sys.A will not be                      %
%           performed, default = true                                                        %
%                                                                                            %
%           sys.baseline baseline of calcium measurements, default value is mean of          %
%           anything within 3*std of noise from minimum of traces                            %
%                                                                                            %
%           sys.confidence confidence in detected spikes, default: 90 %                      %
%
%           sys.fs: sampling frequency for spindles; minimum spindle frequency is set to 8Hz %
%           and maximum is set to be 16 Hz                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written by:                                                                                %
% Abbas Kazemipour, University of Maryland, College Park,                                    %
%                   Janelia Research Campus         Last Update: November,      1, 2016      %
%                                                                                            %
%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(sys,'Order'); sys.Order = 1; end;                                                %
if ~isfield(sys,'A'); theta = aryule(sys.y',sys.Order);                                      % 
    sys.A = -theta(:,2:end);   end; %sys.EMFlag = false;                                     %
if ~isfield(sys,'confidence'); sys.confidence = 0.90; end                                    %
[n,T] = size(sys.y); % number of measurements per frame (n) and number of time frames (T)    %
if ~isfield(sys,'C'); p = n; sys.C = eye(p);                                                 %
else p = size(sys.C,2);  end                             % State-Space dimension (p);        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: The number of iterations is set to be 5 as in most cases this is                     %
% enough for convergence, can easily use convergence criteria (in the main for loop below),  % 
% which is avoided for code simplicity and readability                                       %
maxNumIters = 5;                                                                             %
if isfield(sys,'maxNumIters'); maxNumIters = sys.maxNumIters; end                            %
                                                                                             %
% The parametere epsilon of IRLS should not be chosen too large or too small                 %
% Theoretical results of AR estimation theory suggest very small epsilon lead to             %
% bad estimates of the Model parameteres, whereas large epsilon undesirably                  %
% smoothens the traces, a suggested range is epsilon ~ 1e-6 to 1e-10                         %
epsilon = 1e-15;                                                                             %
                                                                                             %
% lambda can be chosen to be different for different neurons based on the                    %
% estimates of the noise level (modify the code accordingly please)                          %
% lambda is scaled using theoretical guarantees of LASSO                                     %
if isequal(n,p); lambda = sys.lambda * sqrt(T*log(p*T)/n);                                   %
else lambda = sys.lambda * sqrt(log(p*T)/n); end                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate measurement variances                                                            %
if ~isfield(sys,'R'); sys.R = FCSS_calc_var(sys.y);                                               %
%disp('ROI variances calculated from data');    
end                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the baseline as the mean of value 3*std of noise above the minimum of traces           %
%                                                                                            %
if isfield(sys,'baseline'); baseline =  sys.baseline;                                        %
else baseline = FCSS_calc_base(sys);                                                              %
end;                                                                                         %
sys.y = sys.y - repmat(baseline,1,T) ;                                                       %
if ~isfield(sys,'resetRFlag'); sys.resetRFlag = false; end;                                  %
if ~isfield(sys,'EMFlag'); sys.EMFlag = true; end;                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[sys.Q, W] = FCSS_calc_Q(sys.y,sys.A,epsilon,sys.C);                                              %
sys.Q = sys.Q/lambda;                                                                        %
%%%%%%%%%%%%%%%%%%%%% Main IRLS Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iters =1:maxNumIters                                                                     %
    %% Filter Data                                                                           %
    filtered_sys  = FCSS_Kalman_Filter(sys);                                                      %
    %% Smoothen Data                                                                         %
    sys_smoothed  = FCSS_Kalman_Smoother(filtered_sys);                                           %
    %% Update State Transition Matrix via EM Algorithm                                       %
    if sys.EMFlag                                                                            %
        sys.A = FCSS_Expectation_Maximization( sys_smoothed, W );                                 %
        sys_smoothed.A = sys.A;	                                                             %
    end                                                                                      %
    sys = sys_smoothed;                                                                      %
    %% Update the State Covariance Matrix                                                    %
    [sys.Q,W] = FCSS_calc_Q(sys.X_smoothed(1:sys.Order:end,:),sys.A,epsilon,sys.C);               %
    sys.Q = sys.Q/lambda;                                                                    %
%%%%%%%%%%%%%%%%%%%% Update Estimates of the Noise Variances  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     sys.y = sys.X_smoothed(1:sys.Order:end,:);                                               %
    if sys.resetRFlag && n==p                                                                %
    sys.R = diag( var(sys.y-downsample(sys.X_smoothed,sys.Order),[],2) );                    %
    end                                                                                      %
    %% Replace the initial conditions with the new estimates                                 %
    % Default: x0 = 0; Sig0 = I;                                                             %
    sys.x0 = sys.X_smoothed(:,1);                                                            %
    sys.Sig0 = sys.Sigma_smoothed(:,:,1);                                                    %
end                                                                                          %
                                                                                             %
if sys.Order ==2                                                                             %
sys_smoothed.X_smoothed = downsample(sys_smoothed.X_smoothed,2);                             %
end                                                                                          %
if isequal(n,p)                                                                              %
sys_smoothed.X_smoothed = sys_smoothed.X_smoothed + repmat(baseline,1,T) ;                   %
end                                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[sys_smoothed.spikes, sys_smoothed.decon, sys_smoothed.decon2] = FCSS_Find_Spikes(sys_smoothed,sys.confidence);        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%% 

function filtered_sys = FCSS_Kalman_Filter(sys)

%% Kalman Filter for time-varying dynamics
% x(t) = A(t) x(t-1) + w(t),  w(t) ~ N(0,Q(t))
% y(t) = C(t) x(t)   + v(t),  v(t) ~ N(0,R(t))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: C should be given in its original form as written above                                      %
% If AR(2) model is being used A should be given in the modified form A = [a1 a2; 1 0];              %            
% If AR(2) model is being used the diagFlagA variable is set to 1 as                                 % 
% default to make parallel processing possible                                                       %
% Q should be the modified form also   Q~ = [Q 0; 0 0];                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If system order is not specified default option is gonna be AR(1)                                  %
if ~isfield(sys,'Order'); sys.Order = 1; disp('Autoregressive(1) model'); end;                       %
                                                                                                     %
% States (Neurons) are assumed independent by default                                                %
diagFlagA = true;  % Independent states                                                              %
diagFlagR = true;  % Independent Measurement Noises                                                  %
diagFlagQ = true;  % Independent Gaussians                                                           %
                                                                                                     %
y = sys.y; [n,T] = size(y);                                                                          %
                                                                                                     %
p = size(sys.C,2);                                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If Neurons are independent (i.e. diagFlagA = 1) and working in the                                 % 
% denoising regime (i.e. diagFlagC = 1) huge parallel processing may be achieved                     %
                                                                                                     %
diagFlagC = isdiag(sys.C);                                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% More Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    C = sys.C;                                                                                       %
    if sys.Order == 2                                                                                %
    C = upsample(C',sys.Order)';                                                                     %
    end                                                                                              %
                                                                                                     %
    A = FCSS_calc_transition_matrix(sys.A,p,sys.Order);                                                   %
                                                                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ismatrix(sys.Q); diagFlagQ = isdiag(sys.Q); sys.Q = repmat(sys.Q,1,1,T); end                      %
if p ==1                                                                                             %
Q = reshape(upsample(sys.Q,sys.Order^2),[sys.Order sys.Order, T]);                                   %    
else                                                                                                 %
Q = upsample(sys.Q,sys.Order);                                                                       %
Q = upsample(permute(Q,[2 1 3]),sys.Order);                                                          %
end                                                                                                  %
%%%%%%%%%%%%%  Optional: Make measurement noise time-varying %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if ismatrix(sys.R); diagFlagR = isdiag(sys.R); sys.R = repmat(sys.R,1,1,T); end                    %
R = sys.R;                                                                                           %
if size(R,2)<p; error('R should be diagonal'); end                                                   %
                                                                                                     %
%%%%%%%%%%%%%  Initial Estimates of the States Mean (x0) and Covariance (Sig0)  %%%%%%%%%%%%%%%%%%%%%%
% Note: These may or may not get updated in the IRLS after each Smoothing iteration                  %
                                                                                                     %
if ~isfield(sys,'x0'); sys.x0 = zeros(p*sys.Order,1); end;                                           %
x0 = sys.x0;                                                                                         %
                                                                                                     %
if ~isfield(sys,'Sig0'); sys.Sig0 = eye(p*sys.Order)*mean(diag(R)); end;                             %
Sig0 = sys.Sig0;                                                                                     %       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                                                                                     %
Xt = zeros(p*sys.Order,T);                                                                           %
Xtt = zeros(p*sys.Order,T);                                                                          %
Sigt = zeros(p*sys.Order,p*sys.Order,T);                                                             %
Sigmatt = zeros(p*sys.Order,p*sys.Order,T);                                                          %
                                                                                                     %
%% Filtering                                                                                         %
       for t=1:T                                                                                     %
            %% Prediction Step                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: Since A is block diagonal under a lot of denoising regimes,                                  % 
% the following multiplications could also be implemented in parallel                                %
                                                                                                     %
            if t==1                                                                                  %
            Xt(:,t) = A*x0;                                                                          %
            Sigt(:,:,t) = A*Sig0*A'+Q(:,:,t);                                                        %
            else                                                                                     %
            Xt(:,t)  = A*Xtt(:,t-1);                                                                 %
            Sigt(:,:,t) = A*Sigmatt(:,:,t-1)*A'+Q(:,:,t);                                            %
            end                                                                                      %  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            %% Update Step
            
            % Kalman Gain (Semi-Optimized for parallel processing)
                        Kt = (Sigt(:,:,t)*C')/(C*Sigt(:,:,t)*C'+R);
                        
%             switch sys.Order
%                 case 1
%             if diagFlagC && diagFlagA
%                 Kt  = diag( diag(Sigt(:,:,t)).*diag(C)./(diag(Sigt(:,:,t)).*diag(C).^2 + diag(R)) );
%             else
%                 Kt = (Sigt(:,:,t)*C')/(C*Sigt(:,:,t)*C'+R(:,:,t));
%             end
%                 case 2
%                     Kt =[];
%             if diagFlagC && diagFlagA
%                 for i=1:p
%                 Kti = Sigt(2*i-1:2*i,2*i-1:2*i,t)*C(i,2*i-1:2*i)'/( C(i,2*i-1:2*i)*Sigt(2*i-1:2*i,2*i-1:2*i,t)*C(i,2*i-1:2*i)' + R(i,i) );
%                 Kt = blkdiag(Kt,Kti);
%                 end
%             else
%                 Kt = (Sigt(:,:,t)*C')/(C*Sigt(:,:,t)*C'+R);
%             end
%             
%             end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Update Step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: in the denoising regime Kt is block-diagonal and parallel                              %
% processing could potentially make the following multiplications faster                       %
            Xtt(:,t) = Xt(:,t) +Kt*(y(:,t) - C*Xt(:,t));                                       %
            Sigmatt(:,:,t) = Sigt(:,:,t) - Kt*(C*Sigt(:,:,t)*C'+R)*Kt';                        %
       end                                                                                     %
                                                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Writing the Outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Only those outputs required for (possible) parallel processing (Flags)                       %
% and Smoothing are given, prediction estimates can also be given if needed                    %
filtered_sys = sys;                                                                            %
filtered_sys.Xtt = Xtt;                                                                        %
filtered_sys.Sigmatt = Sigmatt;                                                                %
% filtered_sys.Xt = Xt;                                                                        %
% filtered_sys.Sigt = Sigt;                                                                    %
%% Flags                                                                                       %
filtered_sys.diagFlagA = diagFlagA;                                                            %
filtered_sys.diagFlagQ = diagFlagQ;                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function sys_smoothed  = FCSS_Kalman_Smoother( sys )

% Kalman Smoother for variable
% x_t = A x_{t-1} + w_t
% y_t = C x_t + z_t
% Note: The smoother outputs the augmented estimated vectors if the system
% order is 2, need to downsample by a factor of 2
% For time-varying dynamics simply need to change A -> A(t), C -> C(t) and R -> R(t)
% See Kalman_Filter.m for comments on how to do this

%% Initialization        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = sys.y;                                                              %
[n,T] = size(y);                                                        %    
p = size(sys.C,2);                                                      %
                                                                        %
A = FCSS_calc_transition_matrix(sys.A,p,sys.Order);                          %
% A = sys.A;                                                              %
                                                                        %
if p ==1                                                                %
Q = reshape(upsample(sys.Q,sys.Order^2),[sys.Order sys.Order, T]);      %    
else                                                                    %
Q = upsample(sys.Q,sys.Order);                                          %
Q = upsample(permute(Q,[2 1 3]),sys.Order);                             %
end                                                                     %
                                                                        %
Xtt = sys.Xtt;                                                          %        
Sigmatt = sys.Sigmatt;                                                  %    
Sigma_st = zeros(size(Sigmatt));                                        %
                                                                        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Smoothing
    %%%%%%%%%%%%%%%%%%%%% Smoother Initialization %%%%%%%%%%%%%%%%%%%%%%%
    X_smoothed(:,T) = Xtt(:,T);                                         %
    Sigma_smoothed(:,:,T) = Sigmatt(:,:,T);                             %
                                                                        %
    % Calculate Sigma_st(:,:,T) (Required for EM Step)                  %
    t = T;                                                              %
    Sigt = A*Sigmatt(:,:,t)*A'+Q(:,:,t);                                %
        if sys.diagFlagA && sys.diagFlagQ                               %
            St = diag( diag(A).*diag(Sigmatt(:,:,t))./diag(Sigt));      %
        else                                                            %
            St = A*Sigmatt(:,:,t)/Sigt;                                 %
        end                                                             %
    Sigma_st(:,:,t) = St*Sigmatt(:,:,t);                                %
                                                                        %
    %%%%%%%%%%%%%%%%%%%%%%% Backward Iterations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                                                                                                  %
    for t=T-1:-1:1                                                                                                %
        Sigt = A*Sigmatt(:,:,t)*A'+Q(:,:,t);                                                                      %  '
        
        % Smoothing Gain (Semi-Optimized)                                                                         %
                St = A*Sigmatt(:,:,t)/Sigt;                                                                       % 
%         switch sys.Order                                                                                          %
%             case 1                                                                                                %
%                 if sys.diagFlagA && sys.diagFlagQ                                                                 %
%                         St = diag( diag(A).*diag(Sigmatt(:,:,t))./diag(Sigt));                                    %
%                 else                                                                                              %
%                         St = A*Sigmatt(:,:,t)/Sigt;                                                               %
%                 end                                                                                               %
%             case 2                                                                                                %
%                         St = [];                                                                                  %
%                 if sys.diagFlagA && sys.diagFlagQ                                                                 %
%                     for i=1:p                                                                                     %
%                         Sti = A(2*i-1:2*i,2*i-1:2*i)*Sigmatt(2*i-1:2*i,2*i-1:2*i,t)/Sigt(2*i-1:2*i,2*i-1:2*i);    %
%                         St = blkdiag(St,Sti);                                                                     %
%                     end                                                                                           %
%                 else                                                                                              %
%                         St = A(:,:,t)*Sigmatt(:,:,t)/Sigt;                                                        %
%                 end                                                                                               %
%         end                                                                                                       %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Update Step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                                                                                                  %
                        X_smoothed(:,t) = Xtt(:,t) + St*(X_smoothed(:,t+1)-A*Xtt(:,t));                           %
                        Sigma_smoothed(:,:,t) = Sigmatt(:,:,t)+St *(Sigma_smoothed(:,:,t+1)-Sigt)*St';            %
                        Sigma_st(:,:,t) = St*Sigma_smoothed(:,:,t);                                               %
    end                                                                                                           %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Write Outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                                                                                                  %
sys_smoothed = sys;                                                                                               %
sys_smoothed.X_smoothed = X_smoothed;                                                                             %
sys_smoothed.Sigma_smoothed = Sigma_smoothed;                                                                     %
sys_smoothed.Sigma_st = Sigma_st;                                                                                 %
sys_smoothed.diagFlagA = sys.diagFlagA;                                                                           %
sys_smoothed.diagFlagQ = sys.diagFlagQ;                                                                           %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

%% 

function merged = FCSS_merge_spikes(spikes)
spikes(end) = 0;
T = length(spikes);
merged = zeros(size(spikes));
t = 1;
while t <T+1
    index = t;
    while spikes(index) ~= 0
    index = index+1;
    end
    merged(t) = sum(spikes(t:index));
    t = index+1;
end
end














