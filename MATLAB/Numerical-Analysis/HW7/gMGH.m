%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function g = gMGH(x,nprob,n,m)
% gradient evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function g = fMGH(x,nprob,n,m)

[r, J] = vecMGH(x,nprob,n,m,3);
g = J' * r;
