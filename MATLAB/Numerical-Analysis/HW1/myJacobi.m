function [x, info, rres] = myJacobi(A,b,maxit,tol)

    iter = 1;
    N = size(A);
    m = N(1);
    n = N(2);
    x = zeros(n,1);
    rres = [1];
    b_norm = norm(b,2);
    D_inv = 1./diag(A);
    C = D_inv.*b;
    
    while (iter <= maxit) && (rres(iter) > tol)   
        Ax = A*x;
        x = x - D_inv.*Ax + C;
        rres(end+1) = norm(b - Ax,2)/norm(b,2);
        iter = iter + 1;
    end
    if iter >= maxit
        info = sprintf('Maxit reached');
    else
        info = sprintf('Converged at iteration %d', iter-1);
    end
    
end
    
    