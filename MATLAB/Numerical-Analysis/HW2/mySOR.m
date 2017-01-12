function [x, iter, info] = mySOR(A,b,omega,maxit,tol)

    iter = 1;
    N = size(A);
    m = N(1);
    n = N(2);
    x = zeros(n,1);
    rres = 1;
    b_norm = norm(b,2);
    D = diag(diag(A));
    U = triu(A, 1);
    L = tril(A, -1);
    DL = D + omega*L;
    DU = omega*U + (omega-1)*D;
    C = DL\(omega*b);
    
    while (iter <= maxit) && (rres > tol)   
        x = C - (DL)\(DU*x);
        rres = norm(b - A*x,2)/norm(b,2);
        iter = iter + 1;
    end
    if iter >= maxit
        info = sprintf('Maxit reached');
    else
        info = sprintf('Converged at iteration %d', iter-1);
    end
    
end
    
    