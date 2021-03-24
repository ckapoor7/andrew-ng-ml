function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
temp = theta;
temp(1) = 0;
hypothesis = sigmoid(X*theta);
J = (1/m)*sum(-y.*log(hypothesis) - (1-y).*log(1-hypothesis)) + (lambda/(2*m))*sum(temp.^2);
grad = (1/m)*(X)'*(hypothesis-y) + (lambda/m)*temp; %Vectorised gradient
grad = grad(:);

end
