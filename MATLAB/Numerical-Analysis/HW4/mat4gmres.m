function A = mat4gmres(n,fname,varargin);

A = feval(fname,n,varargin{:});


%%%%%%%%%%%%%%%%%%%%%%%%%
function A = mat1(n,p,t);
m = p+1;
e = ones(n,1);
B = spdiags([e -e],-1:2:1,m,m);
B(1,2) = 0; B(2,1) = t; B(1,m) = t;
A = spdiags([e e],-1:0,n,n);
A(1:m,1:m) = B; A(m+1,m) = eps;
%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%
function A = mat2(n,p,t);
e = ones(n,1);
A = spdiags(e,-1,n,n);
A(1,p+1) = 1; 
A(p+2,p+1) = 0;
A(p+2,n) = 1;
A(1,:) = A(1,:) + t;
%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%
function A = mat3(n,p,t);
e = ones(n,1); p = 10;
A = spdiags(e,0,n,n) + spdiags(-t*e,-1,n,n);
A(1,1) = 0.001; A(p+2,p+1) = eps;
%%%%%%%%%%%%%%%%%%%%%%%
