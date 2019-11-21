function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
predictions = X*theta;
distance_sum = sum((predictions - y).^2);

J = (1/(2*m))*distance_sum;

% Calculate the regularization
squre_theta = theta.^2;
sq_sum = sum(squre_theta) - squre_theta(1);
reg = (lambda/(2*m)) * sq_sum;

% Return the cost with regularizition
J = J + reg;



% Compute the scalars for the derivatives
scalars = predictions .- y;

% Compute the grad matrix
grad = (1/m)*(X')*scalars;
tmp = grad(1);
% Computing the regularization of the gradient
grad_reg = (lambda/m)*theta;
  
  % Add the regularized values to each partial derivative
grad = grad + grad_reg;
  % Keep theta 0 as the regular value
grad(1) = tmp;
% =========================================================================

grad = grad(:);

end
