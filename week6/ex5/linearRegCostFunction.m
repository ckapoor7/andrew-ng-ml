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
hypothesis = X*theta;
J = (1/(2*m)) * (hypothesis-y)' * (hypothesis-y) + (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta)); %Skip the bias value

new_theta = theta;
new_theta(1) = 0; %Make the bias value 0 (for conformant matrices)

grad = (1/m)*(hypothesis-y)'*X + (lambda/m)*(new_theta)'; %Vectorized...


grad = grad(:);

end
