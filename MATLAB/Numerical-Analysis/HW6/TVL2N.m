function [f,g] = TVL2N(u,h,t,m,beta)

% Compute the TVL2 function (with pertubation beta): 
%          f = TV(u) + .5*t||u-h||^2
% and its gradient g, where u = vec(U) and U is m x m.

if nargin < 5; beta = .01; end
zr = zeros(1,m);
zc = zeros(m,1);
U = reshape(u,m,m);
Ux = diff([zc U],1,2);
Uy = diff([zr; U]);

fidelity = u - h;
U = sqrt(Ux.^2 + Uy.^2 + beta);
f =  sum(U(:)) + .5*t*norm(fidelity)^2;

if nargout < 2; return; end

Ux = Ux./U; 
Uy = Uy./U;
U = -diff([Ux zc],1,2);
U = U - diff([Uy; zr]);
g = U(:) + t*fidelity;
