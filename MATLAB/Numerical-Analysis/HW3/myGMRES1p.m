function [x,iter] = myGMRES1p(A,b,tol,M1,M2,maxit)
    
    switch nargin
        case 3
            maxit = 200;
            M1 = true;
            M2 = true;
        case 4
            M1 = true;
            M2 = true;
        case 5
            maxit = 200;
    end
    
    if M1
        M1 = 1;
    end
    if M2
        M2 = 1;
    end
    
    M = M1*M2;
    M_inv = inv(M);
    A = M_inv*(A);
    b = M_inv*b;
    maxit = 1200;
    [m , n] = size(A);
    x = zeros(n);
    iter = 0;
    b_norm = norm(b);
    rres = b_norm;
    Q = zeros(m,maxit+2);
    H = zeros(maxit+2, maxit+1);
    e1 = zeros(maxit+2);
    e1(1) = 1;
    Q(:,1) = b/b_norm;
    v = zeros(m);
    while (iter <= maxit) && (rres > tol) 
        
        iter = iter + 1;
        v = (A*Q(:,iter));
        for i = 1:iter
            H(i,iter) = dot(Q(:,i),v);
            v = v - (H(i, iter)*Q(:,i));
        end
        H(iter+1, iter) = norm(v);
        if H(iter+1, iter) > tol
            Q(:,iter+1) = v/H(iter+1,iter);

        end
        y_star = mldivide(H(1:iter+2,1:iter+1),e1(1:iter+2)');
        
        x = Q(:,1:iter+1)*(b_norm*y_star);
        rres = norm(b - A*x,2)/b_norm;
    end
     
    
