function [x,iter] = MyGN(func,x,tol,maxiter,prt,varargin)

	iter = 0;
	[r,J] = feval(func,x,varargin{:});
	norm_g = norm(J'*r);
	p = inv(J'*J)*J'*r;
    f = .5*norm(r,2)^2;
    
    alpha0 = 1;
	C1 = .5;
	P0 = .5;


	function amiho = get_amiho(c1,alph)
		amiho = false;
		[r1,J1] = feval(func, x + alph*p,varargin{:});
        f_alph = .5*norm(r1,2)^2;
		if (f_alph < f + c1*alph*(J'*r)'*p)
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

	alpha = backTrack(P0);


	while iter < maxiter && norm_g > tol

		iter = iter + 1;
		p = -inv(J'*J)*J'*r;
		alpha = backTrack(P0);
		x = x + alpha*p;
		[r,J] = feval(func,x,varargin{:});
        f = .5*norm(r,2)^2;
        if prt
            fprintf('iter: %2i  norm(F) = %7.3e\n',iter,norm_g);
        end
		norm_g = norm(J'*r);
		

	end

end