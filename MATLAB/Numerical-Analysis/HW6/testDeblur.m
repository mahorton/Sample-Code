clear; close all
np = 32;
I = double(checkerboard(np)); 
m = size(I,1);

ferr = @(x,y) norm(x-y)/norm(y);
H = fspecial('gaussian',50,4);
B = imfilter(I,H,'replicate');

subplot(221);imshow(I,[]); title('Original');
xlabel(sprintf('Size: %i x %i',size(I)));
subplot(222);imshow(B,[]); title('Blurred');
xlabel(sprintf('RelErr: %6.3e',ferr(B(:),I(:))));
drawnow

beta = 2e-5;
tol = 5.e-3; maxit = 1000;
u0 = B(:); t = 1.e+4;

if exist('yzCGD','file') && 1
    t0 = cputime;
    [u,iter,nf] = yzCGD(@TVL2B,1,u0,tol,maxit,u0,t,m,H,beta);
    tcpu = cputime - t0;
    [f,g] = TVL2B(u,u0,t,m,H,beta);
    fprintf('yzCGD:  iter %3i, nf = %4i, tcpu = %6.2e\n',iter,nf,tcpu)
    fprintf('yzCGD:  f = %12.6e, norm(g) = %9.3e \n\n',f,norm(g))
    U = reshape(u,m,m);
    subplot(223);imshow(U,[]); title('Restored by yzCGD');
    xlabel(sprintf('RelErr: %6.3e',ferr(U(:),I(:))));
    drawnow
end

if exist('myCGD','file') && 1
    t0 = cputime;
    [u,iter,nf] = myCGD(@TVL2B,1,u0,tol,maxit,u0,t,m,H,beta);
    tcpu = cputime - t0;
    [f,g] = TVL2B(u,u0,t,m,H,beta);
    fprintf('myCGD:  iter %3i, nf = %4i, tcpu = %6.2e\n',iter,nf,tcpu)
    fprintf('myCGD:  f = %12.6e, norm(g) = %9.3e \n\n',f,norm(g))
    U = reshape(u,m,m);
    subplot(224);imshow(U,[]); title('Restored by myCGD');
    xlabel(sprintf('RelErr: %6.3e',ferr(U(:),I(:))));
end