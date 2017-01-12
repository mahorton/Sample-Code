% solve the H-equation (Kelley book, p87)
% via finite-difference Nweton's method

c = 1 - 1e-4;
N = input('(return for default) N = ');
if isempty(N), N = 250; end
fprintf('H-equation: c = %g, N = %i\n\n',c,N)

x0 = ones(N,1);
tol = 1.e-12; 
maxiter = 100;
prt = 1;

t0 = tic;
[x,nrmF,iter] = FdNewton(@yzHeqn,x0,tol,maxiter,prt,c,N);
fprintf('FdNewton using yzHeqn: mean(x) = %.8f\n',mean(x))
fprintf('nrmF %g, iter %i, time %g\n\n',nrmF(end),iter,toc(t0))

if exist('myHeqn','file')
    t0 = tic;
    [x,nrmF,iter] = FdNewton(@myHeqn,x0,tol,maxiter,prt,c,N);
    fprintf('FdNewton using myHeqn: mean(x) = %.8f\n',mean(x))
    fprintf('nrmF %g, iter %i, time %g\n\n',nrmF(end),iter,toc(t0))
end
