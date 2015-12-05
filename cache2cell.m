function cached = cache2cell(data, groupvar, sortMode)
% CACHE2CELL Wrap grouped rows of data into a cell array
%
%   CACHE2CELL(DATA, GROUPVAR, [SORTMODE])
%     DATA is a M by N numeric matrix and GROUPVAR is M by 1 vector 
%     of grouping indices.
%  
%     SORTMODE can be 'ascend' or 'descend' if you want to groups.
%       NOTE: this does NOT ensure data are sorted within groups.
%     
% See also: FINDGROUPS, SORT

% Author: Oleg Komarov, oleg.komarov (at) hotmail (dot) it
% License: BSD 3 clause
% Tested: on R2015b Win7

if size(data,1) ~= numel(groupvar)
    error('cache2cell:sizeMismatch','DATA must have as many rows as elements in the GROUPVAR.')
end
if nargin == 3
    [groupvar,isort] = sort(groupvar,[],sortMode);
    data             = data(isort,:);
end
cached = accumarray(groupvar,(1:size(data))',[],@(x) {data(x,:)});
end