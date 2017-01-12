function [x,nrmF,iter] = myBroyden(F,x0,tol,maxiter,prt,varargin)
    %
    %       input:  F = a function from R^n to R^n
    %              x0 = initial guess in R^n
    %             tol = tolerance > 0
    %         maxiter = maximum iteration number
    %        varargin = paramaters required by F
    %
    %      output:  x = solution (root of F)
    %            nrmF = a vector containing norm of {F(x_k)| k=1:iter}
    %            iter = iteration number used

    
    % This might work!
    [n,m] = size(x0);
    iter = 0;
    B = eye(n);
    nrm = 1000;
    % probably have to replace all these F(x) with feval(F, x, varargin{:})
    x_old = x0;
    f0 = feval(F, x_old, varargin{:});
    f = f0;
    
    nrmF = [];

    while iter < maxiter && nrm > tol
        
        p = B\f;
        x = x_old - p;
        
        f = feval(F, x, varargin{:});
        s = x - x_old;
        y = f - f0;

        B = B + ((y - B*s)*s')/(s'*s);

        iter = iter + 1;
        f0 = f;
        x_old = x;
        
        nrm = norm(f);
        nrmF(end+1) = nrm;
        if prt
            fprintf('iter: %2i  norm(F) = %7.3e\n',iter,nrm);
        end

    end
end

