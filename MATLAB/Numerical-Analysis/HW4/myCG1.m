function [x, rres1, rres2] = myCG1(A,b,tol,steepest)

    if nargin < 4
        steepest = false;
    end
    % he probably wants us to do this...
   
    [m,n] = size(A);
    x = zeros(n,1);
    r = b;
    p = b;
    iter = 0;
    r_inner_old = (r'*r);
    rres1 = [];
    rres2 = [];
    while (iter < 100)
        iter = iter + 1;
        Ap = A*p;
        alpha = (r_inner_old)/(p'*Ap);
        x = x + alpha*p;
        r = r - alpha*Ap;
        r_inner_new = (r'*r);
        rres2(end+1) = sqrt(r_inner_new);
        if steepest
            beta = 0;
            
        else
            beta = r_inner_new/r_inner_old;
            
        end
        rres1(end+1) = norm(b - A*x)/norm(b);
        p = r + beta*p;
        r_inner_old = r_inner_new;
        
        
    end
end
