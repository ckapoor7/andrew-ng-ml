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
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Feed forward
%Layer 1
X = [ones(m,1), X];
z2 =  Theta1 * X';

%Layer 2
a2 = sigmoid(z2);
a2 = [ones(m,1), a2'];
hypothesis = sigmoid(Theta2 * a2');

%Label vectors
y_vec = zeros(num_labels, m); 
for i = 1:m
  y_vec(y(i), i) = 1;
endfor
  
%Compute the cost function
J = (-1/m) * sum(sum(y_vec.*log(hypothesis) + (1-y_vec).*log(1-hypothesis)));

%Regularize the cost function
new_theta1 = Theta1( :,2:size(Theta1, 2)); %Exclude the bias values
new_theta2 = Theta2( :,2:size(Theta2, 2));

%Compute the regularization term
reg_term = (lambda/(2*m)) *(sum(sum(new_theta1.^2)) + sum(sum(new_theta2.^2)));

J += reg_term;

%Back propagation
for t = 1:m

   %Feed forward pass
	  a1 = X(t ,:); %Initialize a1 
          z2 = Theta1 * a1';
          a2 = sigmoid(z2);
          a2 = [1 ; a2]; %Add bias layer
          z3 = Theta2 * a2;
          a3 = sigmoid(z3);

   %Calculate delta values
	  delta_3 = a3 - y_vec( :,t);

   %Step 3
   z2 = [1; z2]; %Add bias layer
   delta_2 = (Theta2' * delta_3).*sigmoidGradient(z2); %Compute delta 2

   %Accumulate gradient errors
   delta_2 = delta_2(2:end); %Skip the bias vector
   Theta2_grad += delta_3 * a2';
   Theta1_grad += delta_2 * a1;

endfor;

%Obtain unregularized gradient
Theta2_grad *= (1/m);
Theta1_grad *= (1/m);



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


