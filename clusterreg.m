function [res, varBhat, Rsq] = clusterreg(y, X, g, model)
% CLUSTERREG OLS regression with one-way or two-way clustered standard errors
%
%   CLUSTERREG(y, X, g)
%       where y (N by 1) is the dependent variable, X is a matrix (N by K)
%       with the regressors and g is a matrix (N by M) of group indices. 
%       In the case of one-way clustering M = 1.
%
%   [RES, VARBHAT, RSQ] = clusterreg(...)
%       RES is a table with:
%           .Estimate
%           .SE         clustered standard errors
%           .tStat      t-statistics
%           .pValue
%       Note: if table() is not available results are concatenated in the
%       order presented above into a K+1 by 3 matrix.
%
%       VARBHAT is a K by K clustered variance-covariance matrix
%   
%       RSQ is the R square of the regression
%
%   References:
%   [1] Cameron, A. C., J. B. Gelbach, and D. L. Miller. "Robust inference
%       with multiway clustering." Journal of Business & Economic
%       Statistics vol.29, no. 2 (2011)

% Originally adapted from code by Ian D. Gow on http://www.people.hbs.edu/igow/GOT/
% Author: Oleg Komarov, oleg.komarov (at) hotmail (dot) it
% License: BSD 3 clause
% Tested: on R2015b Win7

if nargin < 4, model = 'linear'; end
X = x2fx(X,model);

nonnan = ~any(isnan(X),2);
if any(~nonnan)
    X = X(nonnan,:);
    y = y(nonnan,:);
    g = g(nonnan,:);
end

Estimate = X\y;
e        = y - X*Estimate;

% Cluster robust variance on first group of indices
varBhat = clusteredVar(X, e, g(:,1));

if size(g,2) == 2
    % With two clustering indices, the final variance is equal to the sum
    % of the variances clustered by each group, minus the variance from the
    % intersected clusters, i.e. B_hat = VCV_g1 + VCV_g2 - VCV_g12.
    varBhat = varBhat + clusteredVar(X, e, g(:,2)) - clusteredVar(X, e, g);
end

% Calculate stats
SE     = sqrt(diag(varBhat));
tStat  = Estimate./SE;
pValue = 2 * normcdf(-abs(tStat));

% Return the calculated values
try
    res = table(Estimate, SE, tStat, pValue);
catch
    res = [Estimate, SE, tStat, pValue];
end

if nargout == 3
    ybar = mean(y);
    SST  = norm(y-ybar)^2;
    SSE  = norm(e)^2;
    Rsq  = 1 - SSE/SST;
end
end

function varBhat = clusteredVar(X, e, g)
[N, k] = size(X);

if size(g,2) == 2
    [G,trash,glabel] = unique(g, 'rows');
else
    [G,trash,glabel] = unique(g);
end
M = numel(G);

X_g = cache2cell(X,glabel);
e_g = cache2cell(e,glabel);
B   = 0;
for ii = 1:M
    B = B + (X_g{ii}'*e_g{ii})*(e_g{ii}'*X_g{ii});
end

% Calculate cluster-robust variance matrix estimate
q_c     = (N-1)/(N-k)*M/(M-1);
XX      = X'*X;
varBhat = q_c * (XX\B)/XX; % inv(X'*X)*B*inv(X'*X)
end