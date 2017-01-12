function testNLS(factor,tol,CodeRange)

% factor controls ||x0 - x*||

if nargin < 1, factor = 1;         end
if nargin < 2, tol = 1.e-4;        end
if nargin < 3, CodeRange = 3:6;    end
fprintf('\nfactor = %i, tol = %g\n',factor,tol)

fmt1 = 'P# %2i: 2f = %10.4e, |g| = %8.2e, ';
fmt2 = 'iter = %3i, cpu = %5.2e\n';

prt = 0 < 0; 
default = 1; 
maxiter = 500; 
warning off; %#ok<WNOFF>

Code = cell(6,1);
% your codes
Code(1)={'[x,iter]=MyGN(@vecMGH,x,tol,maxiter,prt,nprob,n,m);'};
Code(2)={'[x,iter]=MyLM(@vecMGH,x,tol,maxiter,prt,nprob,n,m);'};
% instructor's codes
Code(3)={'[x,iter]=ZyGN(@vecMGH,x,tol,maxiter,prt,nprob,n,m);'};
Code(4)={'[x,iter]=ZyLM(@vecMGH,x,tol,maxiter,prt,''dog'',nprob,n,m);'};
Code(5)={'[x,iter]=ZyLM(@vecMGH,x,tol,maxiter,prt,''scg'',nprob,n,m);'};
Code(6)={'[x,iter]=ZyLM(@vecMGH,x,tol,maxiter,prt,''exact'',nprob,n,m);'};

%ProbRange = 1:6;               % quick test
ProbRange = [1:26, 28:34];     % full test
%CodeRange = 1:length(Code);    % full test
%CodeRange = [1 3];             % test GN
%CodeRange = [2 4];             % test LM

if prt; t0 = cputime; end
for i = CodeRange;

    callstr = Code{i}; 
    fprintf('\nCode #%i: %s\n',i, callstr);
    Nonconv = []; Niter = 0;
    ti = cputime; 

    for nprob = ProbRange
        [n,m,x] = initMGH(nprob,default,factor);
        tp = cputime;
        eval(callstr);
        if iter==maxiter; Nonconv = [Nonconv nprob]; end %#ok<AGROW>
        time = cputime - tp; Niter = Niter + iter;
        f = fMGH(x,nprob,n,m);
        g = gMGH(x,nprob,n,m);
        fprintf([fmt1 fmt2],nprob,2*f,norm(g),iter,time);
    end
    fprintf(' - Not converged for Prob# %d\n',Nonconv);
    fprintf(' - Number of non-convergence: %i\n',length(Nonconv));
    fprintf(' - Total number of iters: %i\n',Niter);
    fprintf(' - Total CPU time: %g Seconds\n\n',cputime-ti);

end
if prt;
    fprintf('Grand Total CPU time: %g Seconds\n\n',cputime-t0);
end