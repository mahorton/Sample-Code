function [x,iter] = FdNewton(F,x0,tol,maxiter,prt,varargin)
%
% Finite difference Newton: 
% Solve nonlinear systems of equations
%
% Usage: [x,iter] = FdNewton(F,x0,tol,maxiter,p1,p2,...)
%
%       input:  F = a function (or its handle)
%              Fp = derivative (or its handle)
%              x0 = initial guess
%             tol = tolerance
%         maxiter = maximum iteration number
%       p1,p2,... = paramaters required by f(optional)
%
%      output:  x = approximate solution (root of f)
%            iter = iteration number used (optional)
%
x = x0;
for iter = 1:maxiter                              % main loop
      Fx = feval(F,x,varargin{:});                % evaluate F(x)
      nrmFx = norm(Fx);                           % norm of F(x)
      if prt                                      % print out
         fprintf('iter: %2i  norm(F) = %7.3e\n',iter,nrmFx);
      end
      if nrmFx < tol break; end                   % solution found
      Fpx = feval('FdJacobi',F,x,Fx,varargin{:}); % evaluate F(x)
      x = x - Fpx \ Fx;                           % update
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate Jacobian using finite difference %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function J = FdJacobi(Fname, x, Fx, varargin)
n = length(x); eps0 = 1.e-8;

for j = 1:n
 eps1 = eps0 * max(1, abs(x(j)));
 x(j) = x(j) + eps1;
 F = feval(Fname, x, varargin{:});
 x(j) = x(j) - eps1;
 J(:,j) = (F - Fx)/eps1;
end;

%J = (J + J')/2;
