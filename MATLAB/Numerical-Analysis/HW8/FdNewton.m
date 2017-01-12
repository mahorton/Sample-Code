function [x,nrmF,iter] = FdNewton(F,x0,tol,maxiter,prt,varargin)
%
% Finite difference Newton: 
% Solve nonlinear systems of equations
%
% Usage: [x,nrmFx,iter] = FdNewton(F,x0,tol,maxiter,p1,p2,...)
%
%       input:  F = a function (or its handle)
%              Fp = derivative (or its handle)
%              x0 = initial guess
%             tol = tolerance
%         maxiter = maximum iteration number
%       p1,p2,... = paramaters required by f(optional)
%
%      output:  x = solution (root of F)
%            nrmF = norm of F(x_k) (optional)
%            iter = iteration number used (optional)
%
x = x0;
nrmF = zeros(maxiter,1);

for iter = 1:maxiter                              % main loop
      Fx = feval(F,x,varargin{:});                % evaluate F(x)
      nrmFx = norm(Fx);                           % norm of F(x)
      if prt                                      % print out
         fprintf('iter: %2i  norm(F) = %7.3e\n',iter,nrmFx);
      end
      nrmF(iter) = nrmFx; 
      if nrmFx < tol, break; end                  % solution found
      Fpx = feval('FdJacobi',F,x,Fx,varargin{:}); % evaluate F'(x)
      x = x - Fpx \ Fx;                           % update
end
nrmF = nrmF(1:iter);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate Jacobian using finite difference %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = FdJacobi(Fname, x, Fx, varargin)
m = length(Fx); n = length(x); eps0 = 1.e-6;
J = zeros(m,n);

for j = 1:n
 eps1 = eps0 * max(1, abs(x(j)));
 x(j) = x(j) + eps1;
 F = feval(Fname, x, varargin{:});
 x(j) = x(j) - eps1;
 J(:,j) = (F - Fx)/eps1;
end;
end