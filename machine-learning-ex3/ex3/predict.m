function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% % once I figure out Arrays of variables in octave VS vectors :) I'll clean this up
% thetas = [Theta1;Theta2]
% previous_layer = X
%
% for layer = 1:size(thetas, 2)
%   m = size(previous_layer, 1)
%
%   previous_layer = sigmoid([ones(m, 1) previous_layer] * thetas(layer)' )
% endfor


layer_1 = sigmoid([ones(m, 1) X] * Theta1')

m = size(layer_1, 1)
layer_2 = sigmoid([ones(m, 1) layer_1] * Theta2')

[max_value, p] = max(previous_layer, [], 2)











% =========================================================================


end
