function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Calculate the "hypothosies" values
a1 = [ones(m, 1) X];
z2 = a1*(Theta1');
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2*(Theta2');
predictions = sigmoid(z3);

% Turning "y" into KxM matrix. every column is R^K vector that in the i'th place...
% you have 1, and all the other zeros.

new_y = zeros(num_labels,m);
for q = 1:m
  new_y(y(q),q) = 1;
endfor

% Calculating the cost
for i = 1:m
  for k = 1:num_labels
    J = J + ((1/m) * (-new_y(k,i)*log(predictions(i,k)) - (1-new_y(k,i))*log(1-predictions(i,k))));
  endfor
endfor

% Regulaztion
reg = 0;
for j = 1:rows(Theta1)
  for k = 2:columns(Theta1)
    reg = reg + Theta1(j,k)^2;
  endfor
endfor

for j = 1:rows(Theta2)
  for k = 2:columns(Theta2)
    reg = reg + Theta2(j,k)^2;
  endfor
endfor
reg = (lambda/(m*2))*reg;

J = J+reg;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in The  ta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

DELTA_2 = 0;
DELTA_1 = 0;

for t = 1:m
  % Calculating the value of all nodes in a single activation
  a_1 = [1, X(t,:)];
  z_2 = a_1*(Theta1');
  a_2 = [1, sigmoid(z_2)];
  z_3 = a_2*(Theta2');
  a_3 = sigmoid(z_3);
  
  % Calculating delta in each layer
  delta_3 = (a_3') - new_y(:,t);
  new_z2 = [sigmoidGradient(z_2)];
  delta_2 = (((Theta2')*delta_3)(2:end)).*(new_z2');
  
  % Accumilating the gradient
  DELTA_2 = DELTA_2 + delta_3*(a_2);
  DELTA_1 = DELTA_1 + delta_2*(a_1);
endfor

  Theta1_grad = (1/m)*(DELTA_1);
  Theta2_grad = (1/m)*(DELTA_2);


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
reg_theta1_grad = Theta1_grad;
reg_theta1_grad(1,:) = zeros(1,columns(Theta1_grad));
reg_theta1_grad(:,1) = zeros(rows(Theta1_grad),1);
reg_theta1_grad = reg_theta1_grad * (lambda/m);
Theta1_grad = Theta1_grad + reg_theta1_grad;

reg_theta2_grad = Theta2_grad;
reg_theta2_grad(1,:) = zeros(1,columns(Theta2_grad));
reg_theta2_grad(:,1) = zeros(rows(Theta2_grad),1);
reg_theta2_grad = reg_theta2_grad * (lambda/m);
Theta2_grad = Theta2_grad + reg_theta2_grad;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
