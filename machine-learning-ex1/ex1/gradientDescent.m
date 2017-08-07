function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENT Performs gradient descent to learn theta
  %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
  %   taking num_iters gradient steps with learning rate alpha

  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters

      % ====================== YOUR CODE HERE ======================
      % Instructions: Perform a single gradient step on the parameter vector
      %               theta.
      %
      % Hint: While debugging, it can be useful to print out the values
      %       of the cost function (computeCost) and gradient here.
      %
      predictions   = X * theta          % m x 1 vector
      error_vector  = predictions - y    % m x 1 vector
      product       = X' * error_vector  % [n x m] X [m x 1]

      theta = theta - alpha * 1/m * product % scaling of the product


      % ============================================================

      % Save the cost J in every iteration
      J_history(iter) = computeCost(X, y, theta);
  end

  plot(J_history)
end
