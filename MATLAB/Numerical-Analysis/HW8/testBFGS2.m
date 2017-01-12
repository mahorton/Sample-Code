% CAAM 454/554 - test script for BFGS

clear; close all

C = checkerboard(128);
A = C*C';
n = size(A,1); k = 2;
nrmA2 = norm(A,'fro')^2;

tol = 5.e-12;
maxit = 200;

[V,D] = eigs(A,k);
Xs = V*sqrt(D); 
fs = yzlork(Xs(:),n,k,A,nrmA2);

fprintf('n = %i, k = %i, fs = %.8e\n\n',n,k,fs)

rng(0);
x0 = randn(n*k,1);
Solvers(1,:) = 'yzBFGS';
Solvers(2,:) = 'myBFGS';
Funcs = {@yzlork, @mylork};

for j = 1:2
    solver = Solvers(j,:);
    func = Funcs{j};
    if exist(solver,'file')
        t0 = tic;
        [x,iter1,hist] = ...
            eval([solver '(func,x0,tol,maxit,n,k,A,nrmA2)']);
        f = func(x,n,k,A,nrmA2);
        fh = abs(hist(1,:)); gh = hist(2,:);
        fprintf('f - fs = %.8e\n',f-fs)
        fprintf([solver ': time = %8.4e\n\n'],toc(t0));
        figure(1)
        subplot(222-mod(j,2)); semilogy(1:length(fh),fh,'b-*');
        xlabel('Iteration'); ylabel('func')
        title([solver ', x0 = randn']);
        grid on; set(gca,'fontsize',14)
        subplot(224-mod(j,2)); semilogy(1:length(gh),gh,'b-*');
        xlabel('Iteration'); ylabel('||g||_2')
        grid on; set(gca,'fontsize',14)
        figure(2); subplot(122-mod(j,2))
        [X,~]=qr(reshape(x,n,k),0); 
        Im = X*(X'*C); imshow(Im,[]); 
        xlabel([solver ' result']); shg
    end
end
