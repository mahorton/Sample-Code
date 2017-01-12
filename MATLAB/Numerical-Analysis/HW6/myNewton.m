function [x,iter,nf] = myNewton(func, x0, tol, maxit, varargin)
	
	nf = 1;
	iter = 0;
	x = x0;
	[f,g,H] = feval(func,x,varargin{:});
	crit = norm(g)/(1+abs(f));

	while iter <= maxit && crit > tol

		iter = iter + 1;
		x = x - inv(H)*g;
		[f,g,H] = feval(func,x,varargin{:});
		nf = nf + 1;
		crit = norm(g)/(1+abs(f));

	end

end