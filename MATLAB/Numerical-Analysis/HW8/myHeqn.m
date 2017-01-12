function F = myHeqn(x,c,N)

	% Write a matlab function to evaluate the H-equation F(x):
    % where c is a model parameters and N is the number of grid points.

    mu_i = ((1:N) - .5)/N;

    sum=0;
    for j = 1:N
    	mu_j = (j -.5)/N;
    	sum = sum + (mu_i*x(j))./(mu_i + mu_j);
    end
    
    deno = 1 - ((c)/(2*N))*sum;
    %size(deno)
    %size(x)
    F = x-1.0./deno';
end