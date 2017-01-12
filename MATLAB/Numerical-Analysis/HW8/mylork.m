function [f,g] = mylork(x,n,k,A,nrmA2)


	X = reshape(x,[n,k]);
    f = .25*norm(X*X' - A, 'fro')/nrmA2;
    g = ((X*X' - A)*X)/nrmA2;
    g = reshape(g, [n*k, 1]);
end