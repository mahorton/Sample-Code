function [x,iter,hist] = myBFGS(func,x0,tol,maxit,varargin)
	%
	% Quasi-Newton method using inverse BFGS update 
	% (during backtracking, make sure you check y'*s > 0)
	%
	% Usage: 
	%   [x,iter,hist] = myBFGS(func,x0,tol,maxit,varargin);
	%
	% INPUT:
	%       func - matlab function for f and g evaluations
	%         x0 - starting point (column vector)
	%        tol - stopping tolerance for gradient norm
	%      maxit - maximum number of iterations
	%
	% OUTPUT:
	%          x - computed approximate solution
	%       iter - number of iterations taken
	%      hist(1,:) - vector of function values at all iterations
	%      hist(2,:) - vector of gradient norms  at all iterations

    % This might work!'
    [n,m] = size(x0);
    iter = 1;
    H = eye(n);
    [f0,g0] = feval(func, x0, varargin{:});
    x = x0 - g0;
    s = x-x0;
    
    [f,g] = feval(func, x, varargin{:});
    nrm = norm(g);
    hist = [f ; nrm];
	
    alpha0 = 1;
	C1 = .1;
	P0 = .05;
	

    while iter < maxit && nrm > tol
        
        iter = iter + 1;
        y = g-g0;
        
        g0 = g;
        x0 = x;

        ys = y'*s;
        if ys < tol
 
        	break;
        else
            if ys > 0
                pk = 1.0/ys;
                H = H + pk*(-H*y*s' - s*(y'*H) + pk*s*((y'*H)*y)*s' + s*s');
            
            end
            
        end

        
        p = -(H*g);
        alpha = backTrack(P0);
        s = alpha*p;
        x = x + s;
        
        [f,g] = feval(func, x, varargin{:});
        nrm = norm(g);
        hist(:,end+1) = [f;nrm]; 
        

    end


	function amiho = get_amiho(c1,alph)
		amiho = false;
		f_alph = feval(func, x + alph*p, varargin{:});
        if (f_alph < f + c1*alph*g'*p)
			amiho = true;
	    end
	end


	function alph = backTrack(pp)
        alph = alpha0;
		for i = 1:10
			amiho = get_amiho(C1,alph);
			if amiho
				break;
			end
			alph = pp * alph;
		end
    end
    %hist(:,end-1)
	
end