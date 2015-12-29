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

%====================Part 1===========================================
X = [ones(m,1) X];
y_mat = zeros( num_labels, m );
for i = 1 : m
    y2vec = zeros(num_labels, 1);
    y2vec(y(i)) = 1;
    y_mat(:, i) = y2vec;
end
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for i = 1 : m
    a1 = X(i,:)';
    z2 = Theta1 * a1;
    a2 = [1,sigmoid(z2)']';
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    for j = 1 : num_labels
        J = J + (-y_mat(j,i) * log(a3(j)) - (1 - y_mat(j,i)) * log(1 - a3(j))) / m;
    end 
    %backword propogation process
    delta3 = a3 - y_mat(:,i); % delta3 is 10 * 1
    delta2 = Theta2' * delta3 .* (a2 .* (1 - a2)); %delta2 is 26 * 1
    delta2 = delta2(2:end); % remove bias unit
    
    Delta1 = Delta1 + delta2 * a1';
    Delta2 = Delta2 + delta3 * a2';
end

%========================Part 2============================================
temp1 = lambda / m * Theta1(:,2:end);
temp2 = lambda / m * Theta2(:,2:end);
Theta1_grad = [Delta1(:,1)/m Delta1(:,2:end)/m + temp1];
Theta2_grad = [Delta2(:,1)/m Delta2(:,2:end)/m + temp2];
reg1 = 0;
reg2 = 0;
Theta1_rmbias = Theta1(:,2:end);
Theta2_rmbias = Theta2(:,2:end);
for k = 1 : size(Theta1_rmbias,2)
    for j = 1 : size(Theta1_rmbias, 1)
        reg1 = reg1 + (Theta1_rmbias(j,k)) ^ 2; 
    end
end

for k = 1 : size(Theta2_rmbias,2)
    for j = 1 : size(Theta2_rmbias, 1)
        reg2 = reg2 + (Theta2_rmbias(j,k)) ^ 2; 
    end
end
reg = (reg1 + reg2) * lambda / (2 * m);
J = J + reg;




















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
