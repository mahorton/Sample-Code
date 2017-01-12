% CAAM 454/554 - test script for BFGS

func = @rosenbrock_g;
n = 1000;
%n = input('n = ');
tol = 1.e-11;  
maxit = 300;

x01 = 1.2*ones(n,1);
x02 = x01;
x02(1:2:n-1) = -1.2;
x02(2:2:n) = 1;

if exist('yzBFGS','file')
    t0 = cputime;
    disp('Initial Guess 1 ...')
    [~,iter1,hist] = yzBFGS(func,x01,tol,maxit);
    fh1 = hist(1,:); gh1 = hist(2,:);
    disp('Initial Guess 2 ...')
    [~,iter2,hist] = yzBFGS(func,x02,tol,maxit);
    fh2 = hist(1,:); gh2 = hist(2,:);
    fprintf('yzBFGS: total CPU = %8.4e\n\n',cputime-t0);
    figure(1);
    subplot(221); semilogy(1:length(fh1),fh1,'b-*');
    xlabel('Iteration vs. f'); ylabel('func')
    title('yzBFGS, x0 = (1.2, 1.2)'); grid on
    subplot(223); semilogy(1:length(gh1),gh1,'b-*');
    xlabel('Iteration vs. g'); ylabel('||g||_2')
    title('yzBFGS, x0 = (1.2, 1.2)'); grid on
    subplot(222); semilogy(1:length(fh2),fh2,'b-*');
    xlabel('Iteration vs. f'); ylabel('func')
    title('yzBFGS, x0 = (-1.2, 1.0)'); grid on
    subplot(224); semilogy(1:length(gh2),gh2,'b-*');
    xlabel('Iteration vs. g'); ylabel('||g||_2')
    title('yzBFGS, x0 = (-1.2, 1.0)'); grid on
end;

if exist('myBFGS','file')
    t0 = cputime;
    disp('Initial Guess 1 ...')
    [~,iter1,hist] = myBFGS(func,x01,tol,maxit);
    fh1 = hist(1,:); gh1 = hist(2,:);
    disp('Initial Guess 2 ...')
    [~,iter2,hist] = myBFGS(func,x02,tol,maxit);
    fh2 = hist(1,:); gh2 = hist(2,:);
    fprintf('myBFGS: total CPU = %8.4e\n\n',cputime-t0);
    figure(2);
    subplot(221); semilogy(1:length(fh1),fh1,'r-*');
    xlabel('Iteration vs. f'); ylabel('func')
    title('myBFGS, x0 = (1.2, 1.2)'); grid on
    subplot(223); semilogy(1:length(gh1),gh1,'r-*');
    xlabel('Iteration vs. g'); ylabel('||g||_2')
    title('myBFGS, x0 = (1.2, 1.2)'); grid on
    subplot(222); semilogy(1:length(fh2),fh2,'r-*');
    xlabel('Iteration vs. f'); ylabel('func')
    title('myBFGS, x0 = (-1.2, 1.0)'); grid on
    subplot(224); semilogy(1:length(gh2),gh2,'r-*');
    xlabel('Iteration vs. g'); ylabel('||g||_2')
    title('myBFGS, x0 = (-1.2, 1.0)'); grid on
end;