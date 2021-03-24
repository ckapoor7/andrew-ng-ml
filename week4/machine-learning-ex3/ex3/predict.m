function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

%Forward propagation for layer 1
a1 = [ones(m,1), X];
z2 = a1 * Theta1';

%Forward propagation for layer 2
a2 = sigmoid(z2);
a2 = [ones(m,1), a2];
z3 = a2 * Theta2';

res = sigmoid(z3);
[max_prob class_label] = max(res, [], 2); %Return a column vector containing the max value of each row
p = class_label; 


end
