function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

temp = sigmoid(X*theta);
cost = -y.*log(temp) - (1-y).*log(1-temp);
J = sum(cost)/m;

ans = 0;

for i=2:n
ans = ans + theta(i)^2;
end;
ans = (ans*lambda)/(2*m);
J = J + ans;
grad = zeros(size(theta));
temp = temp-y;

grad = (X'*(sigmoid(X*theta)-y))/m;

a = theta;
a = lambda.*a;
a = a/m;
for i=2:n
grad(i) = grad(i) + a(i);
end;







% =============================================================

end
