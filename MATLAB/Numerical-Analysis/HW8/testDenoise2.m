clear; close all
np = 20;
I = checkerboard(np,1);
I = double(I > 0.5);
m = size(I,1);

ferr = @(x,y) norm(x-y)/norm(y);
Noisy = imnoise(I,'gaussian',0,.05);

subplot(221);imshow(I,[]);     title('Original');
xlabel(sprintf('Size: %i x %i',size(I)));
subplot(222);imshow(Noisy,[]); title('Noisy');
xlabel(sprintf('RelErr: %6.3e',ferr(Noisy(:),I(:))));
drawnow

beta = 5e-7;
tol = 1e-3; 
u0 = Noisy(:); 
t = 2;

CGD = @yzCGD;
BFGS = @yzBFGS;

%To run your codes: change 0 to 1
run_my_codes = .5 < 1;

if run_my_codes
    CGD = @myCGD;
    BFGS = @myBFGS;
    disp('Run my Codes Now!')
end

t0 = cputime;
[u,iter,nf] = CGD(@TVL2N,1,u0,tol,2000,u0,t,m,beta);
tcpu = cputime - t0;
[f,g] = TVL2N(u,u0,t,m,beta);
fprintf([char(CGD) ': iter %3i, nf = %4i, tcpu = %6.2e\n'],iter,nf,tcpu)
fprintf([char(CGD) ': f = %12.6e, norm(g) = %9.3e \n\n'],f,norm(g))
U = reshape(u,m,m);
subplot(223);imshow(U,[]); 
title(['Denoised by ' char(CGD)]);
xlabel(sprintf('RelErr: %6.3e',ferr(U(:),I(:))));
drawnow

t0 = cputime;
[u,iter,hist] = BFGS(@TVL2N,u0,tol,300,u0,t,m,beta);
tcpu = cputime - t0;
[f,g] = TVL2N(u,u0,t,m,beta);
fprintf([char(BFGS) ': iter %3i, nf = %4i, tcpu = %6.2e\n'],iter,nf,tcpu)
fprintf([char(BFGS) ': f = %12.6e, norm(g) = %9.3e \n\n'],f,norm(g))
U = reshape(u,m,m);
subplot(224);imshow(U,[]); 
title(['Denoised by ' char(BFGS)]);
xlabel(sprintf('RelErr: %6.3e',ferr(U(:),I(:))));