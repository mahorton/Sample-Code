function [f,g,H] = rosenbrockn(x)
% [f,g,H] = rosenbrockn(x)

n = size(x,1); 
if mod(n,2) ~= 0 error('n must be even'); end
odd = 1:2:n; xodd = x(odd); 
even = odd+1; xeven = x(even); 

t1 = 1 - xodd;
t2 = xeven-xodd.^2;
f = norm(t1)^2 + 10*norm(t2)^2;
if nargout == 1 return; end;

g = zeros(n,1);
g(odd) = 20*t2; g(even) = 20*t2;
g(odd) = g(odd).*(-2*xodd) - 2*t1;
if nargout == 2 return; end;

H = sparse(n,n);
H(odd,odd) = 2*speye(n/2) - 40*sparse(1:n/2,1:n/2,xeven)...
             + 120*sparse(1:n/2,1:n/2,xodd.^2);
H(even,even) = 20*speye(n/2);
H(odd,even) = -40*sparse(1:n/2,1:n/2,xodd);
H(even,odd) = H(odd,even);
