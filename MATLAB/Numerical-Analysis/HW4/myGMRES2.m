function [x,iter] = myGMRES2(A,b,p,tol,maxit)
    
    switch nargin
        case 3
            maxit = 1000;
    end    
    %initialize
    [m , n] = size(A);
    iter = zeros(2);
    b_norm = norm(b);
    x0 = zeros(n,1);
    e1 = zeros(p+2);
    e1(1) = 1;
    beta = b_norm;
    Q = zeros(m,p);
    H = zeros(p+2, p+1);
    Q(:,1) = b/b_norm;
    break_ = true;
    while (iter(1) <= maxit) && break_
        iter(1) = iter(1) + 1;

        iter(2) = 0;
        while (iter(2) < p ) && break_
            
            %arnoldi method to find H and Q
            iter(2) = iter(2) + 1;
            v = A*Q(:,iter(2));
            for i = 1:iter(2)
                H(i,iter(2)) = dot(Q(:,i),v);
                v = v - (H(i, iter(2))*Q(:,i));
            end
            H(iter(2)+1, iter(2)) = norm(v);
            if H(iter(2)+1, iter(2)) > tol
                Q(:,iter(2)+1) = v/H(iter(2)+1,iter(2));
            end
            
            %least squares minimizes H*y - beta*e1
            y_star = H(1:iter(2)+2,1:iter(2)+1)\(beta*e1(1:iter(2)+2)');
            x = x0+Q(:,1:iter(2)+1)*(y_star);
            r = b - A*x;
            r_norm = norm(r);
            rres = r_norm/b_norm;
            if (rres < tol)
                break_ = false;
            end
        end
        x0 = x;
        beta = r_norm;
        Q(:,1) = r/beta;
        
    end
end
