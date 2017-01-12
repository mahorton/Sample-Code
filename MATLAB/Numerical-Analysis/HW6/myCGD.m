function [x,iter,nf] = myCGD(func,method,x0,tol,maxit,varargin)
	
	nf = 1;
	iter = 0;
	x = x0;
	[f,g] = feval(func,x,varargin{:});
	crit = norm(g)/(1+abs(f));
	r = -g;
	pk = r;
    
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

	switch method
		case 1
			while iter <= maxit && crit > tol

				iter = iter + 1;
				x = x + alpha*r;
				[f,g] = feval(func,x,varargin{:});
				nf = nf + 1;
				r = -g;
				alpha = backTrack(P);
				crit = norm(g)/(1+abs(f));
                pk = r;

			end

		case 2
			%CG-PR
			
			while iter <= maxit && crit > tol
				iter = iter + 1;
				x = x + alpha*pk;
				nf = nf + 1;
				[f_new,g_new] = feval(func,x,varargin{:});
				beta = (g_new'*(f_new - f))/(g'*g);
				g = g_new;
				f = f_new;
				r = -g;
				pk = r + beta*pk;
				alpha = backTrack(P);
				
				crit = norm(g)/(1+abs(f));

			end

		case 3
			%CG-PR+
			while iter <= maxit && crit > tol
				iter = iter + 1;
				x = x + alpha*pk;
				nf = nf + 1;
				[f_new,g_new] = feval(func,x,varargin{:});
				beta = max(0,(g_new'*(f_new - f))/(g'*g));
				g = g_new;
				f = f_new;
				r = -g;
				pk = r + beta*pk;
				alpha = backTrack(P);
				
				crit = norm(g)/(1+abs(f));

			end
		case 4
			%CG-FR
			while iter <= maxit && crit > tol
				iter = iter + 1;
				x = x + alpha*pk;
				nf = nf + 1;
				[f,g_new] = feval(func,x,varargin{:});
				beta = (g_new'*g_new)/(g'*g);
				g = g_new;
				r = -g;
				pk = r + beta*pk;
				alpha = backTrack(P);
				
				crit = norm(g)/(1+abs(f));

			end

		otherwise
			while iter <= maxit && crit < tol

				iter = iter + 1;
				x = x + alpha*r;
				[f,g] = feval(func,x,varargin{:});
				nf = nf + 1;
				r = -g;
				alpha = backTrack(P);
				crit = norm(g)/(1+abs(f));

            end
    end
end
