function X = mySolve4X(A, B, tol, maxit)

    [m,n] = size(A);
    function AX = Afun(X) 
        X = reshape(X,[n,n]);
        mat = A*X + X*A;
        AX = reshape(mat, [n*n,1]);
    end
    function xx = pcg_quiet(Afunn, b, tol, maxit)
        xx = pcg(Afunn, b, tol, maxit);
    end
    b = reshape(B,[n*n,1]);
    x = pcg_quiet(@Afun,b,tol,maxit);
    X = reshape(x,[n,n]);
end
    