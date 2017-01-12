%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [r, J] = vecMGH(x,nprob,n,m,option)
% Evaluate r anf J at x for the appropriate test 
% function in the MGH test set, based on nprob.
%    option = 1: r only
%    option = 2: J only
%    option = 3: both
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [r, J] = vecMGH(x,nprob,n,m,option)

global FIRSTIME FNAME;
if nprob == 27 error('prob #27 not here yet'); end

if nargin < 5 option = 3; end
funcname = char(FNAME(nprob));
[r, J] = feval(funcname,n,m,x,option);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%
% Brown badly scaled function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=badscb(n,m,x,option)
% Problem no. 4
% Dimensions -> n=2, m=3              
% Standard starting point -> x=(1,1)  
% Minima -> f=0 at (1e+6,2e-6)        
%                                     
% Revised on 10/22/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = badscb(n,m,x,option)  

fvec = []; J = [];
if (option==1 | option==3)
        fvec = [  x(1)-10^6
                  x(2)-(2e-6)
                  x(1)*x(2)-2  ]  ;
end;        
if (option==2 | option==3)
        J    = [  1      0
                  0      1
                  x(2)   x(1)  ] ;
end;

%

% Powell badley scaled function 
% ----------------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=badscp(n,m,x,option)   
% Problem no. 3
% Dimensions -> n=2, m=2                      
% Standard starting point -> x=(0,1)          
% Minima -> f=0 at (1.098...10-E5,9.106...)   
%                                             
% Revised on 10/22/94 by Madhu Lamba          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = badscp(n,m,x,option)

fvec = []; J = [];
if (option==1 | option==3)
        fvec = [  10^4*x(1)*x(2)-1
                  exp(-x(1))+exp(-x(2))-1.0001  ] ;
end;
if (option==2 | option==3)
        J    = [  10^4*x(2)        10^4*x(1)
                  -exp(-x(1))      -exp(-x(2))  ] ;
end;

%
function [fvec,J] = band(n,m,x,opt)

%******************************************
% Function [fvec, J]= band (n,m,x,opt)
% Broyden banded function   [31]
% Dimensions: n variable,   m=n
% Standard starting point: (-1,...,-1)
% minima of f=0
%
% coded in MATLAB  11/94        plk
% *****************************************   

fvec = []; J = [];
ml=5;
mu=1;
 
J=zeros(m,n);
for i=1:m
        sum=0;
        lb=max(1,i-ml);
        lu=min(n,i+mu);
         
        for j=1:n
           if (j ~= i)
              if((j>=lb)&(j<=lu))
                 sum=sum + x(j)*(1+x(j));
              end;
           end;
        end;


        if((opt==1)|(opt==3))
          fvec(i)=x(i)*(2+5*(x(i)^2))+1-sum;
        end;

        if((opt==2)|(opt==3))
          for j=1:n
            if i==j
               J(i,j)=2+15*(x(i)^2);          
            elseif((j>=lb)&(j<=lu)) 
               J(i,j)=1+2*x(j);
            end;
          end;
        end;
end;                 
fvec=fvec';

if((opt<1)|(opt>3))
   disp('Error: Option value sent to BAND.M is either <1 or >3 ');
end;

function [fvec,J] = bard(n,m,x,opt)
% **************************************************************
% **************************************************************
%  function [fvec,J]= bard(n,m,x,opt)
%  Bard function       [8] 
%  Dimensions  n=3,    m=15
%  Function definition:
%       f(x) = y(i) - [x1 + (u(i) / v(i)x2 + w(i)x3)]
%       where u(i) = i, v(i) = 16-i, w(i) = min(u(i),v(i))
%  Standard starting point at x= (1,1,1)
%  Minima f=8.21487...10^(-3)   and f=17.4286 at (0.8406...,-inf,-inf)
%
%  Revised 10/23/94   PLK
% **************************************************************


fvec = []; J = [];
y    = [.14  .18  .22  .25  .29  .32  .35  .39  .37  .58
        .73  .96  1.34 2.10 4.39   0    0    0    0    0 ]' ;

J=zeros(m,n);

for i = 1:m
     
    u(i) = i;
    v(i) = 16 - i;
    w(i) = min(u(i),v(i));

    if ( (opt ==1) | (opt == 3))
        fvec(i) = y(i)-(x(1)+(u(i)/(v(i)*x(2)+w(i)*x(3))));
    end;

    if ((opt ==2) | (opt ==3))    
        J(i,1) =  -1;
        J(i,2) =  (u(i)*v(i))/((v(i)*x(2)+w(i)*x(3))^2);
        J(i,3) =  (u(i)*w(i))/((v(i)*x(2)+w(i)*x(3))^2);
    end;
    
    
end;
fvec=fvec';
    if ((opt<1)|(opt>3))
        disp('Error: Option value sent to BARD.M is either <1 or >3');
    end;




function [fvec,J] = bd(n,m,x,opt)

%  function [fvec,J] = bd(n,m,x,opt)
%  Brown and Dennis function  [16]
%  Dimensions:  n=4,  m=20
%  Function Definition:
%       f(x)=(x1 + t(i)x2- exp[t(i)])^2 +(x3 + x4sin(t(i))- cos(t(i)))^2
%       where t(i)=i/5
%  Standard starting point (25,5,-5,-1)
%  Minima of f=85822.2... if m=20
%
%  Revised  11/94               PLK
%
fvec = []; J = [];
      two = 2.d0;
      point2 = .2d0;
      x1 = x(1);
      x2 = x(2);
      x3 = x(3);
      x4 = x(4);

     if opt==1
	for i = 1: m
        	ti   = (i)*point2;
        	ei   = exp(ti);
        	si   = sin(ti);
       		ci   = cos(ti);
        	fvec(i) = (x1 + ti*x2 - ei)^2 + (x3 + x4*si - ci)^2;
	end
	fvec=fvec';

     elseif opt==2
	for i=1:m
        	ti = (i)*point2;
        	ei = exp(ti);
        	si = sin(ti);
        	ci = cos(ti);
        	f1 = two*(x1 + ti*x2 - ei);
        	f3 = two*(x3 + x4*si - ci);
        	J( i, 1) = f1;
        	J( i, 2) = f1 * ti;
        	J( i, 3) = f3;
        	J( i, 4) = f3 * si;
	end

     elseif opt==3
	for i=1:m
        	ti = (i)*point2;
        	ei = exp(ti);
        	si = sin(ti);
        	ci = cos(ti);
        	f1 = two*(x1 + ti*x2 - ei);
        	f3 = two*(x3 + x4*si - ci);
        	fvec(i) = (x1 + ti*x2 - ei)^2 + (x3 + x4*si - ci)^2;
        	J( i, 1) = f1;
        	J( i, 2) = f1 * ti;
        	J( i, 3) = f3;
        	J( i, 4) = f3 * si;
	end
	fvec=fvec';

     else 
	error('Error: bd.m - Invalid option')
     end;
% Beale function 
% -------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=beale(n,m,x,option)
% Problem no. 5
% Dimensions -> n=2, m=3              
% Standard starting point -> x=(1,1) 
% Minima -> f=0 at (3,0.5)            
%                                     
% Revised on 10/22/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = beale(n,m,x,option)

fvec = []; J = [];
if (option==1 | option==3)
        fvec = [  1.5-x(1)*(1-x(2))
                  2.25-x(1)*(1-x(2)^2)
                  2.625-x(1)*(1-x(2)^3)  ]; 
end;        
if (option==2 | option==3)
        J    = [  -(1-x(2))      x(1)
                  -(1-x(2)^2)    x(1)*2*x(2)
                  -(1-x(2)^3)    x(1)*3*x(2)^2  ]; 
        
end;
%
function [fvec,J] = biggs(n,m,x,opt)

% ******************************************
%  function [fvec,J] = biggs(n,m,x,opt)
%  Biggs EXP6 function   [18]
%  Dimensions :  n=6,  m=13
%  Standard starting point (1,2,1,1,1,1)
%  Minima of f=5.65565...10^(-3)   if m=13
%            f=0 at (1,10,1,5,4,3)
%
%  Revised  11/94               PLK
% ******************************************
fvec = []; J = [];
J=zeros(m,n);

for i = 1:m
  t(i) = .1*i;
  y(i) = exp(-t(i))-5*exp(-10*t(i))+3*exp(-4*t(i));

 if((opt==1) | ( opt==3))
  fvec(i) = x(3)*exp(-t(i)*x(1))-x(4)*exp(-t(i)*x(2))+x(6)*exp(-t(i)*x(5))-y(i);
 end;

 if((opt==2) | (opt==3))
  J(i,1)  = -t(i)*x(3)*exp(-t(i)*x(1));   
  J(i,2)  =  t(i)*(x(4))*exp(-t(i)*x(2));
  J(i,3)  = exp(-t(i)*x(1));
  J(i,4)  = -exp(-t(i)*x(2));
  J(i,5)  = x(6)*(-t(i))*exp(-t(i)*x(5));
  J(i,6)  = exp(-t(i)*x(5));
 end;

end; 
fvec=fvec';

if((opt<1) | (opt>3))
        disp('Error: Option value sent to BIGGS.M is either <1 or >3');
end;


function [fvec,J] = box(n,m,x,opt)

% *****************************************************************
% *****************************************************************
% Function [fvec,J] = box(n,m,x,opt)
% Box three-dimensional function      [12]
% Dimensions:   n=3     m=10
% Function definition:
%       f(x)= exp[-t(i)x1]-exp[-t(i)x2]-x3[exp[-t(i)]-exp[-10t(i)]]
%       where t(i)=(0.1)i
% Standard Starting Points: (0,10,20)
% Minima of f=0 at (1,10,1), (10,1,-1) and wherever x1=x2 and x3-0
% ******************************************************************

fvec = []; J = [];
for i = 1:m

  t(i) = .1*i;
     
  if((opt ==1) | (opt==3))
     fvec(i) =  exp(-t(i)*x(1))-exp(-t(i)*x(2))-x(3)*(exp(-t(i))-exp(-10*t(i)));
  end;

  if((opt ==2) | (opt ==3))
     J(i,1)  =  -t(i)*exp(-t(i)*x(1));
     J(i,2)  =  t(i)*exp(-t(i)*x(2));
     J(i,3)  = -(exp(-t(i))-exp(-10*t(i)));
  end;

end;
fvec=fvec';

if((opt<1)|(opt>3))
        disp('Error: Option value sent to  BOX.M is either <1 or >3');
end;


% Discrete boundary value function
% -------------------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=bv(n,m,x,option)
% Dimensions -> n=variable, m=n
% Standard starting point -> x=(s(j)) where
%                            s(j)=t(j)*(t(j)-1) where
%                            t(j)=j*h & h=1/(n+1)
% Minima -> f=0 
%                                     
% 12/4/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = bv(n,m,x,option)  

fvec = []; J = [];
J=zeros(m,n);
h=1/(n+1);
for i=1:n
   t(i)=i*h;

   if (option==1 | option==3)
     x(n+1)=0;
     if (i==1)
        fvec(i)=2*x(i)-x(i+1)+(h^2*(x(i)+t(i)+1)^3)/2;    
     elseif (i==n)
        fvec(i)=2*x(i)-x(i-1)+(h^2*(x(i)+t(i)+1)^3)/2;
     else
        fvec(i)=2*x(i)-x(i-1)-x(i+1)+(h^2*(x(i)+t(i)+1)^3)/2;
     end;
   end;

   if (option==2 | option==3)
        J(i,i)=2+h^2/2*3*(x(i)+t(i)+1)^2;
        if (i<n)
           J(i,i+1)=-1;
           J(i+1,i)=-1;
        end;
   end;
end;
 
if (option==1 | option==3)
   fvec=fvec';
end;

%

% Freudenstein and Roth function 
% ------------------------------ 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% function [fvec,J]=froth(n,m,x,option)     
% Problem no. 2
% Dimensions -> n=2, m=2                           
% Standard starting point -> x=(0.5,-2)            
% Minima -> f=0 at (5,4)                           
%           f=48.9842... at (11.41...,-0.8968...)  
%                                                  
% Revised on 10/22/94 by Madhu Lamba               
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = froth(n,m,x,option)

fvec = []; J = [];
if (option==1 | option==3)
        fvec = [ -13+x(1)+((5-x(2))*x(2)-2)*x(2)
                 -29+x(1)+((x(2)+1)*x(2)-14)*x(2) ]; 
end;        
if (option==2 | option==3)
        J    = [ 1       10*x(2)-3*x(2)^2-2
                 1       3*x(2)^2+2*x(2)-14  ] ;
        
end;
%
function [fvec,J] = gauss(n,m,x,opt)

% ****************************************************
% ****************************************************
% function [fvec, J] =gauss(n,m,x,opt)
% Gaussian function [9]      
% Dimensions   n=3,  m=15
% Function definition:
%       f(x)= x(1) exp[-x(2)*[(t(i)-x(3)]^2 / 2]-y(i)
%       where t(i) = (8-i)/2
% Standard starting point at x=(0.4,1,0)
% Minima of f=1.12793...10^(-8)
%
% Revised 10/23/94    PLK
% ****************************************************



fvec = []; J = [];
y    = [.0009  .0044  .0175  .0540  .1295  .2420  .3521  .3989             
            .3521  .2420  .1295  .0540  .0175  .0044  .0009   0   ]' ;

J=zeros(m,n);

for i = 1:m

      t(i) = (8 - i)/2;

      if (opt ==1) | (opt == 3)
        fvec(i) =  x(1)*exp((-x(2)*((t(i)-x(3))^2))/2)-y(i) ;
      end;
      
      if (opt ==2) | (opt == 3) 
        J(i,1)  =  exp((-x(2)*((t(i)-x(3))^2))/2);
        J(i,2)  =  x(1)*((-((t(i)-x(3))^2))/2)*exp((-x(2)*((t(i)-x(3))^2))/2);
        J(i,3)  =  x(1)*x(2)*(t(i)-x(3))*exp((-x(2)*((t(i)-x(3))^2))/2);
      end;
      
end;

fvec=fvec';
 
if ((opt<1) | (opt >3))
        disp('Error: Option value sent to GAUSS.M is either <1 or >3');
end;

%
function [fvec,J] = gulf(n,m,x,opt)

% ***************************************************
% ***************************************************
% function [fvec, J]= gulf(n,m,x,opt)
% Gulf research and development function      [11]
% Dimensions  n=3,  n=<m=<100
% Function definition:
%       f(x)= exp[-(| y(i)mi x(2)|^x(3) / x(1))- t(i)
%       where t(i) = i/100
%       and y(i)=25 +(-50 ln(t(i))^2/3)
% Standard starting point x=(5,2.5,0.15)
% minima of f=0 at (50,25,1.5)
%
% Revised 10/23/94      PLK
% *************************************************** 


fvec = []; J = [];
	zero = 0.d0;
	one = 1.d0;
	point1 = .01d0;
	twnty5 = 25.d0; 
	fifty = 50.d0;

      x1 = x(1);
      x2 = x(2);
      x3 = x(3);
      if (x1 == zero) 
 	disp(' +++ singularity in gulf function evaluation')
      end
      x1inv = one / x1;
      two3rd = 2.d0 / 3.d0;

      for i = 1: m
        ti       = (i)*point1;
        yi       = twnty5 + (-fifty*log(ti)) ^ two3rd;
        ai       = yi - x2;

      	if((opt==1)|(opt==3))
           fvec(i)  = exp(-((abs(ai)^x3)/x1)) - ti;
        end;

        if((opt==2)|(opt==3))
           av = abs(ai);
           bi = av ^ x3;
           ci = bi*x1inv;
           ei = exp(-ci);
           d1 =   ci * x1inv;
           d2 =   x3 * x1inv * av^(x3 - one);
           d3 = - log(av) * ci;
           J(i,1) = d1 * ei;
           if (ai >= zero)  
		J(i,2) =  d2 * ei;
           else
		J(i,2) = -d2 * ei;
           end
           J(i,3) = d3 * ei;
       end
  end;
  fvec = fvec';

if((opt<1)|(opt>3))
       disp('ERRROR: option value sent to GULF.M is either <1 or >3');
end;


function [fvec,J] = helix(n,m,x,opt)

% *******************************************
% *******************************************
% function [fvec, J]= helix(n,m,x,opt)
%
% Helical valley function  [7]
% Dimensions    n=3,   m=3
% Function Definition:
%       f1(x) = 10[x3 - 10*(x1,x2)]
%       f2(x) = 10[((x1)^2 + (x2)^2)^.5 -1]
%       f3(x) = x3
% Standard starting point  x= (-1,0,0)
% Minima of f=0 at (1,0,0)
%
% Revised 10/23/94   PLK
% *********************************************
  
fvec = []; J = [];
if ((opt==1) | (opt ==3))
    if x(1) > 0
	   fvec(1)  =  10*(x(3)-10*((1/(2*pi))*atan(x(2)/x(1))));                                        
    elseif x(1) < 0
	   fvec(1)  =  10*(x(3)-10*((1/(2*pi))*atan(x(2)/x(1))+.5));                       
    end
    fvec(2)  = 10*((x(1)^2+x(2)^2)^.5-1);
    fvec(3)  = x(3);
    fvec=fvec';                   
end;

if ((opt ==2) | (opt == 3))
        J(1,1)   =    (50/pi)*(x(2)/x(1)^2)*(1/(1+(x(2)/x(1))^2));
        J(1,2)   =    (-50/pi)*(1/x(1))*(1/(1+(x(2)/x(1))^2));
        J(1,3)   =    10;

        J(2,1)   =    (10*x(1))/sqrt(x(1)^2+x(2)^2);
        J(2,2)   =    (10*x(2))/sqrt(x(1)^2+x(2)^2);
        J(2,3)   =    0;

        J(3,1)   =    0;
        J(3,2)   =    0;
        J(3,3)   =    1;
end;
 
if ((opt <1) | (opt >3))
        disp('Error: Option value sent to HELIX.M is either <1 or >3');
end;

% Discrete integral equation function
% ----------------------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=ie(n,m,x,option)
% Dimensions -> n=variable, m=n
% Standard starting point -> x=(s(j)) where
%                            s(j)=t(j)*(t(j)-1) where
%                            t(j)=j*h & h=1/(n+1)
% Minima -> f=0 
%                                     
% 12/4/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = ie(n,m,x,option)  

fvec = []; J = [];
J=zeros(m,n);
h=1/(n+1);
for i=1:n
   t(i)=i*h;
end;

for i=1:n

   if (option==1 | option==3)
        x(n+1)=0;
        sum1=0;
        for j=1:i
            sum1=sum1+t(j)*(x(j)+t(j)+1)^3;
        end;
        sum2=0;
        if (n>i) 
          for j=i+1:n
              sum2=sum2+(1-t(j))*(x(j)+t(j)+1)^3;
          end;
        end;
        fvec(i)=x(i)+h*((1-t(i))*sum1+t(i)*sum2)/2;
   end;
 
   if (option==2 | option==3)
      for j=1:n
         if (j<i)
            J(i,j)=3*h/2*(1-t(i))*t(j)*(x(j)+t(j)+1)^2;
         elseif (j>i)
            J(i,j)=3*h/2*t(i)*(1-t(j))*(x(j)+t(j)+1)^2; 
         elseif (j==i)
            J(i,i)=1+3*h/2*(1-t(i))*t(i)*(x(i)+t(i)+1)^2;
         end;
      end;
   end;
end;   
if (option==1 | option==3)
   fvec=fvec';
end;

%

function [fvec,J] = jensam(n,m,x,opt)
% **********************************************
% **********************************************
%
% function [fvec, J]= jensam(n,m,x,opt)
% Jenrich and Sampson function [6]
% Dimensions n=2,   m>=n
% Function definition 
%               f(x)=2+2i-(exp[ix1] + exp[ix2])
% Standard starting point x=(0.3,0.4)
% minima of f=124.362 at x1=x2=0.2578 for m=10
%
% Revised 10/23/94  PLK
% **********************************************

fvec = []; J = [];
J=zeros(m,n);

if ((opt==1) | (opt==3))
        for i=1:m
                fvec(i) =  (2+2*i-(exp(i*x(1))+exp(i*x(2)))) ;
        end
        fvec=fvec';
end;

if ((opt==2) | (opt ==3))
        for i=1:m
                J(i,1)  =  (-i*exp(i*x(1))) ;
                J(i,2)  =  (-i*exp(i*x(2))) ;
        end
end;

if((opt<1)|(opt>3))
  disp(' Error: The option value is either <0 or >3') 
end



function [fvec,J] = kowosb(n,m,x,opt)

% ***********************************************************
% Function [fvec, J]=kowosb(n,m,x,opt)
% Kowalik and Osborne function    [15]
% Dimensions:     n=4   m=11
% Function Definition:
%       f(x)= y(i) - [x1(u^2 +u*x2) / (u^2 + u*x3 + x4)]
% Standard starting point: (0.25,0.39,0.415,0.39)
% Minima of f= 3.07505...10^-4 and
%           f= 1.02734...10^-3  at (inf,-14.07...,-inf,-inf)
%
% Coded in Matlab   October 1994        PLK
% **********************************************************

fvec = []; J = [];
y     = [.1957  .1947  .1735  .1600  .0844  .0627  
         .0456  .0342  .0323  .0235  .0246      0]' ;

u     = [4.0000  2.0000  1.0000  0.5000  0.2500  0.1670
         0.1250  0.1000  0.0833  0.0714  0.0625       0]' ;
         


for i = 1:m
   c1 = u(i)^2 + u(i)*x(2);
   c2 = (u(i)^2 + u(i)*x(3) +x(4));
  
   if((opt==1)|(opt==3))
    fvec(i) =  y(i)-(x(1)*c1)/c2;
   end;

   if((opt==2)|(opt==3))
    J(i,1) =  -c1/c2;
    J(i,2) = (-x(1)*u(i) ) / c2;
    J(i,3) = x(1)*c1*(c2^(-2))*u(i); 
    J(i,4) = x(1)*c1*(c2^(-2));
   end;
end;

fvec=fvec';

if((opt<1)|(opt>3))
        disp('Error: Option value for KOWOSB.M is either <1 or >3');
end;
    
function [fvec,J]= lin(n,m,x,opt)
%Function [fvec,J]= lin(n,m,x,opt)
%Linear function - full rank [32]
%Dimensions: n variable,      m>=n
%Standard starting point: (1,...,1)
%Minima of f=m-n at (-1,...,-1)
%
%Coded in MATLAB   11/94        plk

fvec = []; J = [];
J=zeros(m,n);
for i=1:n

    sum1=sum(x);
        
    if((opt==1)|(opt==3))
        fvec(i)= x(i)-(2/m)*sum1-1;
    end;

    if((opt==2)|(opt==3))
        for j=1:n
           if i==j
                J(i,j)=1-(2/m);
           else J(i,j)=-(2/m);
           end;
        end;
    end;
end;

for i=n+1:m
        
    if((opt==1)|(opt==3))
        fvec(i)= -(2/m)*sum1-1;
    end;

    if((opt==2)|(opt==3))
        for j=1:n
                J(i,j)=-(2/m);
        end;
    end;
end;
       
fvec=fvec';

if((opt<1)|(opt>3))
        disp('Error: Option value sent to LIN.M is either <1 or >3');
end;

function[fvec,J]=lin0(n,m,x,opt)
% ****************************************************
% Function[fvec,J] = lin0(n,m,x,opt)
% Linear function - rank 1 with zero columns and rows  [34] 
% Dimensions: n variable,     m>=n
% Standard starting point: (1,...1)
% Minima of f=(m^2 + 3m - 6)/2(2m - 3)
%
% Coded in MATLAB    11/94      PLK
% *****************************************************

fvec = []; J = [];
J=zeros(m,n);
for i=2:m-1
        sum=0;
        for j=2:n-1
                sum=sum + j*x(j);
        end;
        
        if((opt==1)|(opt==3))
           fvec(i)=(i-1)*sum -1;
        end;

        if((opt==2)|(opt==3))
          for j=2:n-1
             J(1,j)=0;
             J(i,j)=j*(i-1);
             J(m,j)=0;
          end;
        end;
end;
 
if((opt==1)|(opt==3))
    fvec(1)=-1;
    fvec(m)=-1;
end;
if((opt==2)|(opt==3))
    J(1,1)=0;
    J(m,n)=0;
end;
fvec=fvec';

if((opt<1)|(opt>3))
        disp('Error: Option value sent to LIN0.M is either <1 or <3');
end;
function [fvec,J]= lin1(n,m,x,opt)

% **********************************
% Function [fvec,J] = lin1(n,m,x,opt)
% Linear function - rank 1   [33]
% Dimensions:  n variable,    m>=n
% Standard starting point: (1,....,1)
% Minima of f=[(m(m-1))/(2(2m+1))]
%
% Coded in MATLAB    11/94      PLK
% **********************************
fvec = []; J = [];
J=zeros(m,n);
for i = 1:m
   
    sum1=0;
    for j = 1:n
       sum1=sum1 + j*x(j);
    end;

    if((opt==1)|(opt==3))
        fvec(i)= i*sum1 - 1;
    end;

    if((opt==2)|(opt==3))
       
        for j= 1:n
           J(i,j)=i*j;
        end;

    end;
end;
fvec=fvec';

if((opt<1)|(opt>3))
        disp('Error: Option value sent to LIN1.M is either <1 or >3');
end;



function [fvec,J] = meyer(n,m,x,opt)

% ************************************************
% ************************************************             
% function [fvec,J]= meyer(n,m,x,opt)
% Meyer function   [10]
% Dimensions   n=3   m=16
% Function definition:
%       f(x) = x(1)*exp[x(2)/(t(i) + x(3))]-y(i)
%       where t(i)= 45 + 5i
% Standard starting point at x=(0.02,4000,250)
% Minima of f=87.9458...
%
% Revised 10/23/94      PLK
% ************************************************


fvec = []; J = [];
      zero = 0.d0;
      one = 1.d0;

      y = [ 34780.d0
      28610.d0
      23650.d0
      19630.d0
      16370.d0
      13720.d0
      11540.d0
      9744.d0
      8261.d0
      7030.d0
      6005.d0
      5147.d0
      4427.d0
      3820.d0
       3307.d0
       2872.d0 ]' ;
      
      x1 = x(1);
      x2 = x(2);
      x3 = x(3);

    if opt==1
      for i = 1: m
        ti = (45+5*i);
        di = ti + x3;
        ei = exp(x2/di);
        fvec(i) = (x1 * ei) - y(i);
      end
      fvec=fvec';

    elseif opt==2
      for i = 1: m
        ti = (45+5*i);
        di = ti + x3;
        qi = one / di;
        ei = exp(x2*qi);
        si = x1*qi*ei;
        J(i,1) =  ei;
        J(i,2) =  si;
        J(i,3) = -x2*qi*si;
      end

     elseif opt==3
      for i = 1: m
        ti = (45+5*i);
        di = ti + x3;
        qi = one / di;
        ei = exp(x2*qi);
        si = x1*qi*ei;
        fvec(i) = (x1*ei) - y(i);
        J(i,1) =  ei;
        J(i,2) =  si;
        J(i,3) = -x2*qi*si;
      end
      fvec=fvec';

     else
        disp('Error: Option value sent to MEYER.M is either <1 or >3');
     end;


function [fvec,J]  =  osb1(n,m,x,option)

% *******************************************************
% function [fvec,J] = osb1(n,m,x,option)
%  Osborne 1 function   [17]
% Dimensions: n=5 , m=33
% Function Definition:
%       f(x)=y(i)-(x1+x2*exp[-t(i)x4]+x3*exp[-t(i)x5])
%       where t(i)=10(i-1)
% Standard starting point: (0.5,1.5,-1,0.01,0.02)
% Minima of f=5.46489...10^(-5)
%           at (.3754,1.9358,-1.4647,0.01287,0.02212)
%
% Revised  11/94                PLK
% *******************************************************

fvec = []; J = [];
      global FIRSTIME y;

      x1 = x(1);
      x2 = x(2);
      x3 = x(3);
      x4 = x(4);
      x5 = x(5);

if (FIRSTIME)
      y( 1) = 0.844d0;
      y( 2) = 0.908d0;
      y( 3) = 0.932d0;
      y( 4) = 0.936d0;
      y( 5) = 0.925d0;	
      y( 6) = 0.908d0;
      y( 7) = 0.881d0;
      y( 8) = 0.850d0;
      y( 9) = 0.818d0;
      y(10) = 0.784d0;
      y(11) = 0.751d0;
      y(12) = 0.718d0;
      y(13) = 0.685d0;
      y(14) = 0.658d0;
      y(15) = 0.628d0;
      y(16) = 0.603d0;
      y(17) = 0.580d0;
      y(18) = 0.558d0;
      y(19) = 0.538d0;
      y(20) = 0.522d0;
      y(21) = 0.506d0;
      y(22) = 0.490d0;
      y(23) = 0.478d0;
      y(24) = 0.467d0;
      y(25) = 0.457d0;
      y(26) = 0.448d0;
      y(27) = 0.438d0;
      y(28) = 0.431d0;
      y(29) = 0.424d0;
      y(30) = 0.420d0;
      y(31) = 0.414d0;
      y(32) = 0.411d0;
      y(33) = 0.406d0;
      y = y';
      FIRSTIME=0;
end;

      im1 = 0.0d0;
      for i = 1: m
        ti   =  im1*10.d0;
        e4 = exp(-ti*x4);
        e5 = exp(-ti*x5);
        t2 = x2*e4;
        t3 = x3*e5;
	if (option==1 | option==3)
	        fvec(i) = (x1 + t2 + t3) - y(i);
	end
	if (option ==2 | option ==3)
        	J( i, 1) = 1.d0;
        	J( i, 2) =  e4;
        	J( i, 3) =  e5;
        	J( i, 4) = -ti*t2;
        	J( i, 5) = -ti*t3;
	end
        im1 = i;
      end
      fvec = fvec';


% x0 = [.5,1.5,-1,.01,.02]';
%
        
function [fvec,J] = osb2(n,m,x,opt)
% **************************************************************
% **************************************************************
%
%  Function [fvec,J] = osb2(n,m,x,opt)
%  Osborne 2 function      [19]
%  Dimensions:  n=11, m=65
%  Standard starting point: (1.3,0.65,0.7,0.6,3,5,7,2,4.5,5.5)
%  Minima of f=4.01377...10^(-2) 
%  at (1.31,.4315,.6336,.5993,.7539,.9056,1.3651,4.8248,2.3988,
%       4.5689,5.6754)
%
%  Revised   11/94              Plk
% **************************************************************

fvec = []; J = [];
        global FIRSTIME y;

	if (FIRSTIME)
	 y  = [1.366  1.191  1.112  1.013  .991 
        	.885  .831  .847  .786 .725 
         	.746   .679   .608  .655  .616  
             	.606  .602  .626    .651  .724 
             	.649   .649  .694  .644  .624  
             	.661  .612   .558  .533  .495 
               	.500  .423  .395  .375  .372  
            	.391  .396   .405   .428   .429  
              	.523  .562  .607  .653  .672  
           	.708   .633   .668   .645  .632  
              	.591  .559  .597  .625  .739  
              	.710   .729   .720  .636  .581  
              	.428  .292  .162  .098   .054 ]';
	 FIRSTIME=0;
	end; 
            
for i = 1:m
        t(i) = (i-1)/10;
	t09= t(i)-x(9);
	t10= t(i)-x(10);
	t11= t(i)-x(11);
	s09= t09^2;
	s10= t10^2;
	s11= t11^2;
	e1= exp(-t(i)*x(5));
	e2= exp(-s09*x(6));
	e3= exp(-s10*x(7));
	e4=exp(-s11*x(8));

    if((opt==1) | (opt==3))
        fvec(i) = (x(1)*e1 + x(2)*e2 + x(3)*e3 + x(4)*e4)-y(i);
    end;
        
    if((opt==2) | (opt==3))
         r2=x(2)*e2;
         r3=x(3)*e3;
         r4=x(4)*e4;

        J(i,1)=e1;
        J(i,2)=e2;
        J(i,3)=e3;
        J(i,4)=e4;
        J(i,5)=-t(i)*x(1)*e1;
        J(i,6)= -s09*r2;
        J(i,7)= -s10*r3;
	J(i,8)= -s11*r4;
	J(i,9)= 2*t09*x(6)*r2;
	J(i,10)=2*t10*x(7)*r3;
	J(i,11)=2*t11*x(8)*r4;
    end;

end;

fvec=fvec';

if((opt<1) | (opt>3))
        disp('Error: Option value sent to OSB2.M is either <1 or >3');
end;


%x0 = [1.3,.65,.65,.7,.6,3,5,7,2,4.5,5.5]' ;
%


% Penalty I  function
% ------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=pen1(n,m,x,option)
% Dimensions -> n=variable, m=n+1
% Problem no. 23             
% Standard starting point -> x=(s(j)) where 
%                            s(j)=j 
% Minima -> f=2.24997...10^(-5)  if n=4
%           f=7.08765...10^(-5)  if n=10            
%                                     
% 11/21/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = pen1(n,m,x,option)  

fvec = []; J = [];
J=zeros(m,n);
for i=1:n

   if (option==1 | option==3)
        fvec(i)=sqrt(1.e-5)*(x(i)-1);
   end;        

   if (option==2 | option==3)
	J(i,i) = sqrt(1.e-5);
   end;
end; 
   
if (option==1 | option==3)
   sum=0;
   for j=1:n
       sum=sum+x(j)'*x(j);
   end;
   fvec(n+1)=sum-(1/4);
end;

if (option==2 | option==3)
   for j=1:n
        J(n+1,j) = 2*x(j);
   end;
end; 

if (option==1 | option==3)
   fvec=fvec';
end;

%

% Penalty II  function
% ------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=pen2(n,m,x,option)
% Dimensions -> n=variable, m=2*n
% Problem no. 24             
% Standard starting point -> x=(1/2,......,1/2)
% Minima -> f=9.37629...10^(-6)  if n=4
%           f=2.93660...10^(-4)  if n=10            
%                                     
% 11/21/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = pen2(n,m,x,option)  


fvec = []; J = [];
J=zeros(m,n);
if (option==1 | option==3)
   fvec(1)=x(1)-0.2;
end;

if (option==2 | option==3)
   J(1,1)=1;
end;

if(n>=2)

   for i=2:n
     y(i)=exp(i/10)+exp((i-1)/10);

     if (option==1 | option==3)
        fvec(i)=sqrt(1.e-5)*(exp(x(i)/10)+exp(x(i-1)/10)-y(i));
     end;        

     if (option==2 | option==3)
	J(i,i)   = sqrt(1.e-5)*exp(x(i)/10)*(1/10);
        J(i,i-1) = sqrt(1.e-5)*exp(x(i-1)/10)*(1/10);
     end;
   end;
  
   for i=n+1:(2*n-1)
       
     if (option==1 | option==3)
        fvec(i)=sqrt(1.e-5)*(exp(x(i-n+1)/10)-exp(-1/10));
     end;
 
     if (option==2 | option==3)
	J(i,i-n+1) = sqrt(1.e-5)*exp(x(i-n+1)/10)*(1/10);
     end; 
   end;
end;

if (option==1 | option==3) 

    sum=0;
    for j=1:n
        sum=sum+(n-j+1)*x(j)^2;
    end;
    fvec(2*n)=sum-1; 
end;
if (option==2 | option==3)
    for j=1:n
        J(m,j) = (n-j+1)*2*x(j);
    end;
end;
if (option==1 | option==3)
   fvec=fvec';
end;

%

% Rosenbrock function 
% ------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=rose(n,m,x,option)
% Problem no. 1
% Dimensions -> n=2, m=2              
% Standard starting point -> x=(1,1)  
% Minima -> f=0 at (1,1)              
%                                     
% Revised on 10/22/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = rose(n,m,x,option)  
fvec = []; J = [];
if (option==1 | option==3)
        fvec = [  10*(x(2)-x(1)^2)
                  1-x(1)  ] ; 
end;        
if (option==2 | option==3)
        J    = [  -20*x(1)  10
                  -1        0  ] ; 

end;
%
% Extended Rosenbrock function 
% ---------------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=rosex(n,m,x,option)
% Dimensions -> n=variable but even, m=n 
% Problem no. 21            
% Standard starting point -> x=(s(j)) where 
%                            s(2*j-1)=-1.2, 
%                            s(2*j)=1 
% Minima -> f=0 at (1,.....,1)              
%                                     
% 11/21/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = rosex(n,m,x,option)  

fvec = []; J = [];
J=zeros(m,n);

for i=1:m/2

   if (option==1 | option==3)
        fvec(2*i-1)=10*(x(2*i)-x(2*i-1)^2);
        fvec(2*i)=1-x(2*i-1);
   end;        

   if (option==2 | option==3)
        J(2*i-1,2*i-1) = -20*x(2*i-1);
        J(2*i-1,2*i)   = 10; 
        J(2*i,2*i-1)   = -1;
   end;

end;

if (option==1 | option==3)
	fvec=fvec';
end;

 
%
function [fvec,J] = sing(n,m,x,opt)

% ***************************************
% Function [fvec,J]= sing(n,m,x,opt)
% Powell singular function  [13]
% Dimensions:    n=4    m=4
% Function definitions:
%       f1(x)=x1 + 10x2      
%       f2(x)= 5^.5*(x3 - x4)
%       f3(x)= (x2-2x3)^2
%       f4(x)= 10^.5*(x1-x4)^2
% Starting point: (3,-1,0,1)
% Minima of f=0 at the origin
%
% Coded in Matlab  October 31   PLK
% **************************************8


fvec = []; J = [];
if((opt ==1 )|(opt ==3))

        fvec =  [  x(1)+10*x(2)
          	   sqrt(5)*(x(3)-x(4))
          	   (x(2)-2*x(3))^2
                   sqrt(10)*((x(1)-x(4))^2)  ] ;
end;

if((opt==2)|(opt==3))

J    =  [  1                           10       0                            0
           0                            0       sqrt(5)               -sqrt(5)
           0              2*(x(2)-2*x(3))      -4*(x(2)-2*x(3))              0 
           2*sqrt(10)*(x(1)-x(4))       0       0      -2*sqrt(10)*(x(1)-x(4))];

end;
if((opt<1)|(opt>3))
        disp('Error: Option value for SING.M is either <1 or >3');
end;
% Extended Powell Singular function
% --------------------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=singx(n,m,x,option)
% Dimensions -> n=variable but a multiple of 4, m=n             
% Problem no. 22
% Standard starting point -> x=(s(j)) where 
%                            s(4*j-3)=3, 
%                            s(4*j-2)=-1,
%                            s(4*j-1)=0,
%                            s(4*j)=1 
% Minima -> f=0 at the origin.            
%                                     
% 11/21/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = singx(n,m,x,option)  

fvec = []; J = [];
J=zeros(m,n);
for i=1:m/4

   if (option==1 | option==3)
        fvec(4*i-3)=x(4*i-3)+10*(x(4*i-2));
        fvec(4*i-2)=sqrt(5)*(x(4*i-1)-x(4*i));
        fvec(4*i-1)=(x(4*i-2)-2*(x(4*i-1)))^2;
        fvec(4*i)  =sqrt(10)*(x(4*i-3)-x(4*i))^2;
   end;        

   if (option==2 | option==3)
	J(4*i-3,4*i-3) = 1;
        J(4*i-3,4*i-2) = 10;
        J(4*i-2,4*i-1) = sqrt(5);
        J(4*i-2,4*i)   = -sqrt(5);
        J(4*i-1,4*i-2) = 2*x(4*i-2)-4*x(4*i-1);
        J(4*i-1,4*i-1) = 8*x(4*i-1)-4*x(4*i-2);
        J(4*i,4*i-3)   = 2*sqrt(10)*(x(4*i-3)-x(4*i));
        J(4*i,4*i)     = 2*sqrt(10)*(x(4*i)-x(4*i-3));
   end;
end;

if (option==1 | option==3)
   fvec=fvec';
end;

%
% Broyden tridiagonal function
% ---------------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=trid(n,m,x,option)
% Dimensions -> n=variable, m=n
% Problem no. 30
% Standard starting point -> x=(-1,..,-1)
% Minima -> f=0 
%                                     
% 11/21/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = trid(n,m,x,option)  

fvec = []; J = [];
J=zeros(m,n);
for i=1:n

   if (option==1 | option==3)
      x(n+1)=0;
      if (i==1)
         fvec(i)=(3-2*x(i))*x(i)-2*x(i+1)+1;
      elseif (i==n)
         fvec(i)=(3-2*x(i))*x(i)-x(i-1)+1;
      else
         fvec(i)=(3-2*x(i))*x(i)-x(i-1)-2*x(i+1)+1;
      end; 
   end;       

   if (option==2 | option==3)
      J(i,i)=3-4*x(i);
      if (i<n)
         J(i,i+1)=-2;
         J(i+1,i)=-1;
      end;
   end;
end; 

if (option==1 | option==3)
   fvec=fvec';
end;

%

% Trigonometric function
% ---------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=trig(n,m,x,option)
% Problem no. 26
% Dimensions -> n=variable, m=n
% Standard starting point -> x=(1/n,..,1/n)
% Minima -> f=0 
%                                     
% 11/21/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = trig(n,m,x,option)  

fvec = []; J = [];
  zero=0.d0;
  one=1.d0;

  if option==1
      sum1 = zero;
      for i=1:n
        xi   = x(i);
        cxi  = cos(xi);
        sum1  = sum1 + cxi;
        fvec(i) = n + (i)*(one - cxi) - sin(xi);
      end
      fvec=fvec';
      fvec=fvec-sum1;
   
  elseif option==2
      for j=1:n
        xj  = x(j);
        sxj = sin(xj);
        for i=1:n
          J( i, j) = sxj;
        end
        J(j, j) =  (j+1)*sxj - cos(xj);
      end

  elseif option==3
      sum1 = zero;
      for i=1:n
        xi   = x(i);
        cxi  = cos(xi);
        sum1  = sum1 + cxi;
        fvec(i) = n + (i)*(one - cxi) - sin(xi);
      end
      fvec=fvec';
      fvec=fvec-sum1;

      for j=1:n
        xj  = x(j);
        sxj = sin(xj);
        for i=1:n
          J( i, j) = sxj;
        end
        J(j, j) =  (j+1)*sxj - cos(xj);
      end

  else error('Error: trig.m : invalid option')
  end
% Variably Dimensioned function
% ----------------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=vardim(n,m,x,option)
% Dimensions -> n=variable, m=n+2
% Problem no. 25
% Standard starting point -> x=(s(j)) where 
%                            s(j)=1-(j/n) 
% Minima -> f=0 at (1,.....1)
%                                     
% 11/21/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = vardim(n,m,x,option)  

fvec = []; J = [];
J=zeros(m,n);
for i=1:n

   if (option==1 | option==3)
        fvec(i)=x(i)-1;
   end;        

   if (option==2 | option==3)
	J(i,i)=1;
   end;
end 

var_1=0;
for j=1:n
    var_1=var_1+j*(x(j)-1);
end;
if (option==1 | option==3)
        fvec(n+1)=var_1;
        fvec(n+2)=(var_1)^2;
end;

if (option==2 | option==3)
        for j=1:n
 	    J(n+1,j) = j;
            J(n+2,j) = (2*var_1*j);
        end;
end;

if (option==1 | option==3)
   fvec=fvec';
end;

%

% **************************************************
%  Function [fvec,J]= watson(n,m,x,option)
%  Watson function    [20]
%  Dimensions : n=20,  m=31
%  Standard starting point : (0,....,0)
%  Minima of f=2.28767...10^(-3)    if n=6
%            f=1.39976...10^(-6)    if n=9
%            f=4.72238...10^(-10)   if n=12
%	     f=2.48631...10^(-20)   if n=20
%
% Revised  11/94        PLK
% **************************************************


function [fvec,J]= watson(n,m,x,option)
fvec = []; J = [];
for i = 1:29
    
    t(i) = i / 29;
    sum1 = 0;
    for j = 2:n
        sum1 = sum1 + (j-1) *(x(j) * (t(i)^(j-2)));
    end;

    sum2 = 0;
    for j = 1:n
        sum2 = sum2 + x(j) * (t(i)^(j-1));
    end;

  if((option==1) | ( option==3))
    fvec(i) =  sum1-(sum2^2)-1;
  end;

  if ((option==2)|(option==3))
    J(i,1)  =  -(2*sum2);
   
    for j = 2:n
        J(i,j) =  (j-1)*((t(i))^(j-2))-2*sum2*(t(i))^(j-1);
    end;
  end;

end;

if((option==1)|(option==3)) 
  fvec(30) =  x(1);
  fvec(31) =  x(2)-((x(1))^2)-1;
end;

if((option==2)|(option==3))
J(30,1)  =  1;

for r = 2:n
    J(30,r) = 0;
end;



J(31,1)  =  -2*x(1);
J(31,2)  =  1;

for r = 3:n
    J(31,r) = 0;
end;
end;%if statement
 
fvec=fvec';

if((option<1)|(option>3))
        disp('Error: Option value sent to WATSON.M is either <1 or >3');
end;




%
function [fvec,J] = wood(n,m,x,option)

% *********************************************
% *********************************************
%
% Function [fvec,J]=WOOD (n,m,x,option)
% Wood function    [14] 
% Dimensions:     n=4   m=6
% Function Definition:
%       f1(x)= 10(x2 -x1^2)
%       f2(x)= 1 - x1
%       f3(x)= (90)^.5*(x4-x3^2)
%       f4(x)= 1-x3
%       f5(x)= (10)^.5 * (x2+x4-2)
%       f6(x)= (10)^(-.5) * (x2-x4)
% Standard starting point:  (-3,-1,-3,-1)
% Minima of f=0 at (1,1,1,1)
% *********************************************x

fvec = []; J = [];
if((option==1)|(option ==3))

fvec =   [  10*(x(2)-x(1)^2)
            1-x(1)
            sqrt(90)*(x(4)-x(3)^2)
            1-x(3)
            sqrt(10)*(x(2)+x(4)-2)
            (1/sqrt(10))*(x(2)-x(4))  ];
end;

if ((option==2)|(option==3))
J    =   [  -20*x(1)       10        0            0
            -1     	   0         0            0
             0         	   0    -2*sqrt(90)*x(3)  sqrt(90)    
             0             0        -1            0
             0           sqrt(10)    0           sqrt(10)
             0         1/sqrt(10)    0          -1/sqrt(10)  ] ;

end;

if((option<1)|(option>3))
        disp('Error: Option value for WOOD.M is either <1 or >3');
end;
