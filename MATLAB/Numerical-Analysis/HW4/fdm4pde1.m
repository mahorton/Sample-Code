function A = fdm4pde1(m,a,b,c)
%
% sparse matrix of order m^2 resulting 
% from discretizing PDE on unit square:
%
% Lu + a*du/dx + b*du/dy + c*u = f(x,y)  
%        on boundary: u = 0
%
% with the 5-point operator for Laplacian, 
% backward-difference for du/dx and du/dy,
% on an (m+2)-by-(m+2) mesh (where inner
% node values of u are unknown).

h = 1/(m + 1); 
e = ones(m,1); I = speye(m);
S = spdiags([-e 2*e -e], -1:1, m, m);
F = spdiags([-e e], -1:0, m, m);

A = kron(I/h^2,S) + kron(S,I/h^2);
A = A + kron(F ,(a/h)*I);
A = A + kron(F',(b/h)*I);
A = A + kron(I ,    c*I);