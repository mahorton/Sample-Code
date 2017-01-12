function [f,g,H] = Aquartic(x,A,a)

    [m,n] = size(A);
	f = (x'*A*x)^2/4 + a*(sum(x) - n)^2/2;
	g = (x'*A*x)*(A*x) + a*(sum(x) - n);
	H = (A*x)*(A*x)' + (x'*A*x)*A + a;
end