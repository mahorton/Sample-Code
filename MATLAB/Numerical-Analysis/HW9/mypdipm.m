function [x,y,z,iter] = mypdipm(A,b,c,tol,maxit,prt)
	
	[m,n] = size(A);
	iter = 0;
	p = symamd(abs(A) * abs(A)');
	x = ones(n,1);
	z = ones(n,1);
	y = zeros(m,1);
	p_inv = 1:m;
	p_inv(p) = 1:m;
	crit = 1e3;
    epsi = tol*speye(m);
    
	while iter < maxit && crit > tol

		iter = iter + 1;
        xz = x'*z;
		sig = min(.2, 10*(xz)/n);   
		mu = sig*xz/n;           

		r_d = c - A'*y - z;        
		r_p = b - A*x;
		r_c = mu - x.*z; 

        M = bsxfun(@times, A, (x./z)')*A';
		rhs = r_p + bsxfun(@times,A,(1./z)')*(x.*r_d - r_c);

		R = chol(M(p,p) + epsi);
		dy = R\(R'\rhs(p));        
		dy = dy(p_inv);
		dz = r_d - A'*dy;          
		dx = (r_c - x.*dz)./z;

		neg_dx_ind = (dx < 0);
		xx = x(neg_dx_ind);
        dxx = dx(neg_dx_ind);
        alpha_x = min(-xx./dxx);

		neg_dz_ind = (dz < 0);
		zz = z(neg_dz_ind);
        dzz = dz(neg_dz_ind);
        alpha_z= min(-zz./dzz);
        
        tau = max(.9995, 1-10*(x'*z)/n);
        
        x = x + min(tau*alpha_x,1)*dx;
		y = y + min(tau*alpha_z,1)*dy;
        z = z + min(tau*alpha_z,1)*dz;

        rp_norm = norm(r_p)/(1 + norm(b));
        rd_norm = norm(r_d)/(1 + norm(c));
        gap = abs(c'*x - b'*y)/(1 + abs(b'*y));
		crit = rp_norm + rd_norm + gap;

		if prt
			% print iter, relative primal and dual residual norms, and relative duality gap
			% example "iter 1: [primal dual gap] = [9.69e+02 1.61e+03 5.13e+05]"
            fprintf('iter %i:  [primal dual gap] = [%e%e%e]\n',iter,rp_norm, rd_norm, gap);
        end
    end
end