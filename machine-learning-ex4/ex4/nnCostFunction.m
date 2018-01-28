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

% Feedforward the values - vectorized
a0 = [ones(m, 1) X];

z1 = a0 * Theta1';
a1 = [ones(m, 1) sigmoid(z1)];

z2 = a1 * Theta2';
H = a2 = sigmoid(z2);



% Calculate the J for the results
for idx = 1:m
  predict_output   = a2(idx,:);
  y_output         = zeros(num_labels, 1);
  y_output(y(idx)) = 1;

  for k_idx = 1:num_labels
    J += (-1 * y_output(k_idx)) * log(predict_output(k_idx)) - (1 - y_output(k_idx)) * log(1 - predict_output(k_idx));
  endfor
endfor

J *= (1/m);

% Regularization
J += (lambda/(2 * m)) * ...
        ( ...
            sum(sum(Theta1(:, 2:end) .^ 2)) + ...
            sum(sum(Theta2(:, 2:end) .^ 2)) ...
        );


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

% fprintf('\nBackpropagation loop... \n');
Y = repmat(1:num_labels, m, 1) == y;

delta3 = H - Y;
delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidGradient(z1);

DELTA2 = delta3'*a1;
DELTA1 = delta2'*a0;

%
% gradient_1 = 0;
% gradient_2 = 0;
%
% for idx = 1:m
%   y_output         = zeros(num_labels, 1);
%   y_output(y(idx)) = 1;
%
%   fprintf('\nCalculating delta_3 for %i... \n', idx);
%   delta_3 = a2(idx, :)' .- y_output
%
%   fprintf('\nCalculating sigmoidGradient for z2 for %i... \n', idx);
%   sigmoidGradientZ2 = sigmoidGradient(z2(idx, :));
%
%   size(Theta2' * delta_3)
%   size(sigmoidGradientZ2)
%
%   delta_2 = ((Theta2') * ...
%               delta_3 ) .* ...
%               sigmoidGradientZ2
%
%
%   fprintf('\nCalculating gradients for %i... \n', idx);
%   gradient_2 += delta_3 * a2(idx, :)
%   gradient_1 += delta_2(2:end, :) * a1(idx, :)
%
% endfor

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




reg2 = (lambda / m) .* Theta2; reg2(:,1) = 0;
reg1 = (lambda / m) .* Theta1; reg1(:,1) = 0;
Theta2_grad = (DELTA2 ./ m) + reg2;
Theta1_grad = (DELTA1 ./ m) + reg1;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
