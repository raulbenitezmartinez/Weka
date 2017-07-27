n = 80;
real_mu = 0;
real_sigma = 1;
data = normrnd(real_mu, real_sigma, 1e3, 1);

mu = linspace(-5, 5, n);
sigma = linspace(0, 5, n);
logl = zeros(n, n);
for i=1:n
    for j=1:n
        logl(j, i)=log(normlike([mu(i), sigma(j)], data));
    end
end

[x y] = meshgrid(mu, sigma);
surf(x, y, logl, 'LineStyle', 'none');
xlabel('Mean');
ylabel('Standard Deviation');
zlabel('Log Log Likelihood');