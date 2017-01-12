clear; close all
np = 64;
I = double(checkerboard(np)); 
m = size(I,1);

ferr = @(x,y) norm(x-y)/norm(y);
Noisy = imnoise(I,'gaussian',0,.1);

subplot(221);imshow(I,[]);     title('Original');
xlabel(sprintf('Size: %i x %i',size(I)));
subplot(222);imshow(Noisy,[]); title('Noisy');
xlabel(sprintf('RelErr: %6.3e',ferr(Noisy(:),I(:))));
drawnow

beta = 5e-5;
tol = 5e-5; maxit = 300;
u0 = Noisy(:); t = 2;

if exist('yzCGD','file') && 1
    t0 = cputime;
    [u,iter,nf] = yzCGD(@TVL2N,1,u0,tol,maxit,u0,t,m,beta);
    tcpu = cputime - t0;
    [f,g] = TVL2N(u,u0,t,m,beta);
    fprintf('yzCGD:  iter %3i, nf = %4i, tcpu = %6.2e\n',iter,nf,tcpu)
    fprintf('yzCGD:  f = %12.6e, norm(g) = %9.3e \n\n',f,norm(g))
    U = reshape(u,m,m);
    subplot(223);imshow(U,[]); title('Denoised by yzCGD');
    xlabel(sprintf('RelErr: %6.3e',ferr(U(:),I(:))));
    drawnow
end

if exist('myCGD','file') && 1
    t0 = cputime;
    [u,iter,nf] = myCGD(@TVL2N,1,u0,tol,maxit,u0,t,m,beta);
    tcpu = cputime - t0;
    [f,g] = TVL2N(u,u0,t,m,beta);
    fprintf('myCGD:  iter %3i, nf = %4i, tcpu = %6.2e\n',iter,nf,tcpu)
    fprintf('myCGD:  f = %12.6e, norm(g) = %9.3e \n\n',f,norm(g))
    U = reshape(u,m,m);
    subplot(224);imshow(U,[]); title('Denoised by myCGD');
    xlabel(sprintf('RelErr: %6.3e',ferr(U(:),I(:))));
end