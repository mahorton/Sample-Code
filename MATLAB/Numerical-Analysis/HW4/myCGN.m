function [x, iter] = myCGN(A,b,tol)

    % he probably wants us to do this...
    A = A'*A;
    b = A'*b;
    
    [m,n] = size(A);
    x = zeros(n,1);
    r = b;
    p = b;
    maxit = n*n;
    iter = 0;
    r_inner_old = (r'*r);
    while (iter < maxit)
        iter = iter + 1;
        Ap = A*p;
        alpha = (r_inner_old)/(p'*Ap);
        x = x + alpha*p;
        r = r - alpha*Ap;
        r_inner_new = (r'*r);
        if sqrt(r_inner_new)<tol
            break
        end
        beta = r_inner_new/r_inner_old;
        p = r + beta*p;
        r_inner_old = r_inner_new;
        
    end
end


