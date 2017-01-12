function [x,iter] = myGMRES1(A,b,tol, maxit)
    
    switch nargin
        case 3
            maxit = 1000;
    end    
    
    [m , n] = size(A);
    x = zeros(n);
    iter = 0;
    b_norm = norm(b);
    rres = b_norm;
    Q = zeros(m,maxit+1);
    H = zeros(maxit+1, maxit);
    e1 = zeros(maxit+1);
    e1(1) = 1;
    Q(:,1) = b/b_norm;
    v = zeros(m);
    while (iter <= maxit) && (rres > tol) 
        
        iter = iter + 1;
        v = A*Q(:,iter);
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
        