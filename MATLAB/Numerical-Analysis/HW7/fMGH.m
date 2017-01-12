%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function f = fMGH(x,nprob,n,m)
% function evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = fMGH(x,nprob,n,m)

[r, J] = vecMGH(x,nprob,n,m,1);
f = 0.5 * (r' * r);
