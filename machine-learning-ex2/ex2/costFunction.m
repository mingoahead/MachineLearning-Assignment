function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
temp = sigmoid( X * theta);
%====================<1.interation style>==========================
%left = 0;right = 0;
%for i = 1 : m
%    left = left + y(i) * log(temp(i));
%    right = right + (1 - y(i)) * log(1 - temp(i));
%end
%J = (-1.0 / m) * (left + right);
%==================================================================

%=====================<2.vectorization style>======================
J = (-1.0 / m) * sum(y .* log(temp) + (1 - y) .* log(1 - temp));
%==================================================================

for j = 1 : n
 %   colsum = 0;
%    for i = 1 : m
%        colsum = colsum + (temp(i) - y(i)) * X(i,j);
%    end
    grad(j) = (1.0 / m) * sum((temp - y) .* X(:,j));
%    grad(j) = colsum;
end
    










% =============================================================

end
