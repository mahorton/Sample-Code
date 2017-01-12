function [x,nrmF,iter] = myLBroyden(F,x0,tol,maxiter,L,prt,varargin)
    %
    %       input:  F = a function from R^n to R^n
    %              x0 = initial guess in R^n
    %             tol = tolerance > 0
    %         maxiter = maximum iteration number
    %               L = number of pairs of saved update vectors
    %             prt = switch for screen printout (1 on, 0 off)
    %        varargin = paramaters required by F
    %
    %      output:  x = solution (root of F)
    %            nrmF = a vector containing norm of {F(x_k)| k=1:iter}
    %            iter = iteration number used
    iter = 0;
    f = feval(F, x0, varargin{:});
    x = x0;
    nrm = norm(f);
    l = 0;
    nrmF=[];
    C = [];
    D = [];
    
    function g = B_inv(x,C,D)
        [n,m] = size(D);
        g = x;
        if m > 0
            for i = 1:m
                g = g + C(:,i)*(D(:,i)'*g);
            end
        end
    end

    while iter < maxiter && nrm > tol

        s = -B_inv(f,C,D);
        x = x + s;
        f = feval(F, x, varargin{:});
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if l == L
            C = [];
            D = [];
            l = 0;
        else
            y = B_inv(f,C,D);
            ss = s'*s;
            c = (-ss/(ss + s'*y))*y;
            d = s/ss;
            C(:,end+1) = c;
            D(:,end+1) = d;
            l = l+1;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        nrm = norm(f);
        iter = iter+1;
        nrmF(end+1) = nrm;
        if prt
            fprintf('iter: %2i  norm(F) = %7.3e\n',iter,nrm);
        end
    end
end