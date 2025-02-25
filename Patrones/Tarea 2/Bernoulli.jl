using Plots, LaTeXStrings;

f(x, y) = x * log(x/y) + (1-x) * log((1-x)/(1-y))


x = 0.01:0.01:0.99
y = 0.01:0.01:0.99
z = @. f(x', y)
plt1 = plot(x, y, z, xlims=[0,1], ylims=[0,1], st=:surface, color=:plasma, size=[800,600], xlabel = L"\theta_1", ylabel = L"\theta_2", zlabel = L"d_{KL}(P,Q)", title="KL divergence between two Bernoulli distributions")
savefig(plt1,"ex1.png")