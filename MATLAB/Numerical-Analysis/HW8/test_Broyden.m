% solve the H-equation (Kelley book, p87)
%      via Broyden's method

c = 1 - 1e-4;
N = input('(return for default) N = ');
if isempty(N), N = 2500; end
fprintf('H-equation: c = %g, N = %i\n\n',c,N)

x0 = ones(N,1);
tol = 1.e-12; 
maxiter = 100;
prt = 1;

t0 = tic;
[x,nrmF,iter] = yzBroyden(@yzHeqn,x0,tol,maxiter,prt,c,N);
fprintf('yzBroyden using yzHeqn: mean(x) = %.8f\n',mean(x))
fprintf('nrmF %g, iter %i, time %g\n\n',nrmF(end),iter,toc(t0))
semilogy(1:iter,nrmF)

if exist('myBroyden','file')
    t0 = tic;
    [x,nrmF,iter] = myBroyden(@myHeqn,x0,tol,maxiter,prt,c,N);
    fprintf('myBroyden using myHeqn: mean(x) = %.8f\n',mean(x))
    fprintf('nrmF %g, iter %i, time %g\n\n',nrmF(end),iter,toc(t0))
    hold on, semilogy(1:iter,nrmF), hold off
end

semilogy(1:iter,nrmF)
