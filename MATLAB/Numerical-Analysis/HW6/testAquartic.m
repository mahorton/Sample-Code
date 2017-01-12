n = 1000;
%n = 2*n;
A = gallery('minij',n);
a = 1e-1;
x0 = ones(n,1);

if exist('yzNewton.p','file') && 1
    fprintf('\n--- Run yzNewton ---\n')
    tol = 1e-12;
    t0 = cputime;
    [x,iter,nf] = yzNewton(@yzAquartic,x0,tol,100,A,a);
    tcpu = cputime - t0;
    [f,g] = Aquartic(x,A,a);
    fprintf('yzNewton:  iter %3i, nf = %4i, tcpu = %6.2e\n\n',...
        iter,nf,tcpu)

    tol = 2e-3;
    for method = 1:4
        t0 = cputime;
        [x,iter,nf] = yzCGD(@Aquartic,method,x0,tol,1000,A,a);
        tcpu = cputime - t0;
        [f,g] = Aquartic(x,A,a);
        fprintf('method %i:  iter %3i, nf = %4i, tcpu = %6.2e\n\n',...
            method,iter,nf,tcpu)
    end
end

if exist('myNewton.m','file') && 1
    fprintf('\n--- Run myNewton ---\n')
    tol = 1e-12;
    t0 = cputime;
    [x,iter,nf] = myNewton(@Aquartic,x0,tol,100,A,a);
    tcpu = cputime - t0;
    [f,g] = Aquartic(x,A,a);
    fprintf('myNewton:  iter %3i, nf = %4i, tcpu = %6.2e\n\n',...
        iter,nf,tcpu)

    tol = 2e-3;
    for method = 1:4
        t0 = cputime;
        [x,iter,nf] = myCGD(@Aquartic,method,x0,tol,1000,A,a);
        tcpu = cputime - t0;
        [f,g] = Aquartic(x,A,a);
        fprintf('method %i:  iter %3i, nf = %4i, tcpu = %6.2e\n\n',...
            method,iter,nf,tcpu)
    end
end