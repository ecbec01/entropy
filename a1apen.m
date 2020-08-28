function varargout = a1apen(x, varargin)
% A1APEN Area 1 of Approximate Entropy
%
%   VALUE = A1APEN(X) computes the a1ApEn of a vector X.
%
%   VALUES = A1APEN(X, WINDOW_SIZE)computes the moving a1ApEn of a
%   vector X, using non-overlapping windows with WINDOW_SIZE points.
%
%   VALUES = A1APEN(X, WINDOW_SIZE, STEP)computes the moving a1ApEn of a
%   vector X, using windows with WINDOW_SIZE points and setting the
%   interval between adjacent epochs equal to STEP.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VERIFICATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check number of input and output arguments
narginchk(1, 3);
nargoutchk(0,1);

% Check if the first input argument is valid
validateattributes(x, {'single', 'double'}, ...
    {'column', 'nonnan', 'real', 'finite', 'nonsparse'});

% Check if optional input arguments are valid
if nargin >= 2
    for i = 1:nargin - 1
        validateattributes(varargin{i}, {'single', 'double'}, ...
            {'nonnan', 'real', 'finite', 'nonsparse', ...
            'positive', 'scalar'});
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Reshape data vector into a matrix if optional inputs are provided.
% The resultant M x N matrix corresponds to data segmented into
% N epochs with M samples each (moving window analysis).
switch nargin
    case 2
        x = buffer(x, varargin{1}, varargin{1} - 1);
        x = x(:, varargin{1}:varargin{1}:end);
    case 3
        x = buffer(x, varargin{1}, varargin{1} - 1);
        x = x(:, varargin{1}:varargin{2}:end);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE A1APEN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute a1ApEn for each epoch
a1 = nan(size(x, 2), 1);
for i = 1:size(x, 2)
    [r, r_norm] = find_radius(x(:, i));
    ap = apen(x(:, i), r);
    a1(i) = trapz(r_norm, ap);
end
varargout{1} = a1;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Return a vector of tolerances to be used in the a1ApEn calculation
function [r, r_norm] = find_radius(x)

distance = diff(sort(x));
distance = sort(distance);
max_distance = sum(distance);

sub_tol_Da = distance;
sub_tol_Da(distance == 0) = [];

min_distance = min(sub_tol_Da);
amplitude = max_distance - min_distance;

sub_tol_tamanho = length(sub_tol_Da);

if sub_tol_tamanho <= 300
    r = zeros(sub_tol_tamanho + 1, 1);
    for j = 2:sub_tol_tamanho
        r(j) = sum(sub_tol_Da(1 : j - 1));
    end
    r(sub_tol_tamanho + 1) = max_distance;

else
    r1 = arrayfun(@(x) sum(sub_tol_Da(1:x)), 1:50);
    ref2 = sum(sub_tol_Da(1:51));
    ref1 = ref2 / amplitude;

    if ref1 <= 0.02
        pontos = 10 * (1 + ceil((5 - (100 * ref1))));
        p5 = 0.05 * amplitude;
        p35 = 0.35 * amplitude;
        r3 = (ref2 : ((p5 - ref2) / pontos) : p5);
        r4 = (p5 : ((p35 - p5) / 200) : p35);
        r5 = (p35 : ((max_distance - p35) / 100) : max_distance);
        r2 = [r3, r4(2:end), r5(2:end)];

    elseif ref1 > 0.02 && ref1 < 0.1
        pontos = 10 * (1 + ceil((11 - (100 * ref1))));
        p11 = 0.11 * amplitude;
        p35 = 0.35 * amplitude;
        r3 = (ref2 : ((p11 - ref2) / pontos) : p11);
        r4 = (p11 : ((p35 - p11) / 150) : p35);
        r5 = (p35 : ((max_distance - p35) / 100) : max_distance);
        r2 = [r3, r4(2:end), r5(2:end)];

    elseif ref1 >= 0.1 && ref1 <= 0.2
        p35 = 0.35 * amplitude;
        r3 = (ref2 : ((p35 - ref2) / 150) : p35);
        r4 = (p35 : ((max_distance - p35) / 100) : max_distance);
        r2 = [r3, r4(2:end)];

    elseif ref1 > 0.2 && ref1 <= 0.35
        p50 = 0.5 * amplitude;
        r3 = (ref2 : ((p50 - ref2) / 150) : p50);
        r4 = (p50 : ((max_distance - p50) / 50) : max_distance);
        r2 = [r3, r4(2:end)];

    elseif ref1 > 0.35 && ref1 <= 0.5
        r2 = (ref2 : ((max_distance - ref2) / 100) : max_distance);

    elseif ref1 > 0.5
        r2 = (ref2 : ((max_distance - ref2) / 50) : max_distance);

    end

    r = [0; r1'; r2'];

end

r_norm = r ./ max_distance;

end

% Compute the approximate entropy of a vector x, given a vector r of
% tolerances
function varargout = apen(x, r)

% Set lag and dimension.
lag = 1;
dim = 1;

% Construct phase space for original dimension and extended dimension
ps1  = predmaint.internal.NonlinearFeatures.getPhaseSpace(x, lag, dim);
ps2 = predmaint.internal.NonlinearFeatures.getPhaseSpace(x, lag, dim + 1);

% Compute approximate entropy
varargout{1} = phi(ps1, r) - phi(ps2, r);

end

% Compute phi, used in the apen calculation
function varargout = phi(ps, r)

% Initialize
N = size(ps, 1);
C = zeros(length(r), N);

% Calculate number of within-range points
for i = 1:N
    I = max(abs(ps(i, :) - ps), [], 2) <= r';
    C(:, i) = sum(I) / N;
end

% Compute phi
logC = log(C);
logC(isinf(logC)) = [];
varargout{1} = mean(logC, 2);

end
