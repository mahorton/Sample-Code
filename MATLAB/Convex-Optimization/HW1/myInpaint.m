function X = myInpaint(Xh, Omega)

    [m,n] = size(Xh);
    m = length(Omega);
    
    B = ones(n,2);
    B(:,2) = -1;
    B(1,2) = 0;
    d = [0,1];
    D = spdiags(B,d,n-1,n);
    
    E1 = kron(speye(n), D);
    E2 = kron(D, speye(n));
    
    E = [E1;E2];
     
    S = speye(n^2);
    S = S(Omega,:);

    Aeq = [E -speye(2*n*(n-1)) speye(2*n*(n-1)); S sparse(m,2*n*(n-1)) sparse(m,2*n*(n-1))];

    X_omega = reshape(Xh, [n^2, 1]);
    X_omega = Xh(Omega);

    f = [zeros(1,n^2) ones(1,4*n*(n-1))];
    beq = [zeros(1,2*n*(n-1)) X_omega];
    A = [];
    b = [];
    lb = zeros(1,n^2+4*n*(n-1));
    z = linprog(f, A, b, Aeq, beq, lb);

    X = z(1:n^2);
    X = reshape(X, [n,n]);

    
