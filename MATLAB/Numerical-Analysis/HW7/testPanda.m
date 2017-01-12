%close all

% test panda1 or panda2
im = input('Image 1 or 2: ');
if isempty(im) || im <= 1;
     load Panda1; im = 1;
else load Panda2; im = 2;
end

M0 = Panda;
[m,n] = size(M0);
nrm0 = norm(M0,'fro')^2;
fprintf('image size: m = %i, n = %i\n',m,n);

% target rank: e.g., 32 to 128
k = input('target rank (return for default): k = ');
if isempty(k); k = 64*im; end

tol = 1e-2;
maxit = 100;
fprintf('m = %i, n = %i, k = %i, tol = %.e\n\n',m,n,k,tol)

A = M0*M0';
nrmA = norm(A,'fro');
A = A / nrmA;
opts.tol = tol;
f = @(M).25*norm(M-M0,'fro')^2/nrm0;
r = @(U)norm(A*U-U*(U'*A*U),'fro');

% Dimension reduction by SVDS
tic;
[U1,S1,V1] = svds(M0,k); 
M1 = U1*S1*V1';
fprintf(' SVDs  ... '); toc

% Dimension reduction by yzGNes
tic; [X2,iter2] = yzGNex(A,k,tol,maxit);
[U2,~] = qr(X2,0);
M2 = U2*(U2'*M0);
fprintf('yzGNex ... '); toc

subplot(221); imshow(M0,[]);
xlabel(['Original Rank ' num2str(min(m,n))])
subplot(222); imshow(M1,[]);
xlabel(['SVDs Rank ' num2str(k)])
subplot(223); imshow(M2,[]);
xlabel(['yzGNex Rank ' num2str(k)])
shg

% Dimension reduction by myGNex
if exist('myGNex','file')
    tic; [X3,iter3] = myGNex(A,k,tol,maxit);
    [U3,~] = qr(X3,0);
    M3 = U3*(U3'*M0);
    fprintf('myGNex ... '); toc
    subplot(224); imshow(M3,[]);
    xlabel(['myGNex Rank ' num2str(k)])
end

% more info
if 0 < 1
    fprintf('\nSVDs    : f = %.4e, r = %.4e\n',f(M1),r(U1));
    fprintf('iter %3i: f = %.4e, r = %.4e\n',iter2,f(M2),r(U2));
    if exist('M3','var')
        fprintf('iter %3i: f = %.4e, r = %.4e\n\n',iter3,f(M3),r(U3));
    end
end
