function [x,iter] = myGN(func,x,tol,maxiter,prt,varargin)

	iter = 0;
	x = x0;
	[r,J] = feval(func,x,varargin{:});
	norm_g = norm(J'*r);
	

    alpha0 = 1;
	C1 = .5;
	P = .5;


	function amiho = get_amiho(c1,alph)
		amiho = false;
		f_alph = feval(func, x + alph*pk,varargin{:});
		nf = nf + 1;
		if (f_alph < f + c1*alph*g'*pk)
			amiho = true;
	    end
	end


	function alph = backTrack(p)
        alph = alpha0;
		for i = 1:10
			amiho = get_amiho(C1,alph);
			if amiho
				break;
			end
			alph = p * alph;
		end
	end

	alpha = backTrack(P);


	while iter < maxit && norm_g > tol

		iter = iter + 1;
		p = inv(J'*J)*J'*r;
		alpha = backTrack(p);
		x = x + alpha*p;      %minus?
		[r,J] = feval(func,x,varargin{:});
		norm_g = norm(J'*r);
		

	end

end