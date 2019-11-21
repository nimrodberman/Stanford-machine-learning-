function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for i = 1:size(X) % Iterate over every training exampke
  tmp = X(i,:);
  result = 1;
  norm = sum(((tmp - centroids(1,:)).^2));
  for k = 2:K % Iterate over every centroid
    % if the norm of the centroid is smaller set it to be the centroid
    next_norm = sum(((tmp - centroids(k,:)).^2)); 
    if (norm > next_norm)
        result = k;
        norm = next_norm;
    endif
  endfor
  % insert the centroid to the idx table
  idx(i) = result;
endfor






% =============================================================

end

