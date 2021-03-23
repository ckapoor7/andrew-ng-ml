function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
theta_vec = theta;
theta_vec(1) = 0; %Account for counting from the 1st element
grad = zeros(size(theta));
hypothesis = sigmoid(X*theta);
J = (-1/m)*sum(y.*log(hypothesis) + (1-y).*log(1-hypothesis)) + (lambda/(2*m))*sum(theta_vec.^2); %Regularized cost function
grad = (1/m)*(hypothesis-y)'*X + (lambda/m)*theta_vec';

end
