function [X,iter] = myGNex(A,k,tol,maxit)

	iter = 0;
	[m,n] = size(A);
	X = eye(n);
	X = X(:,1:k);
    count = 0;
    old_nrm = 1e6;
    
    
	while iter < maxit
		
		xTx = (X'*X);
		iter = iter + 1;
		G = X*xTx - A*X;
        diff = G/xTx;
        X = X - diff;
        
        nrm = norm(diff, 'fro');
        if nrm-old_nrm < tol
            if count == 1
                break;
            else
                count = 1;
            end
        else
            count = 0;
        end
        old_nrm = nrm;
	end

end