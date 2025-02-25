using Plots, LaTeXStrings;

f(μ, σ) = - log(σ) + 0.5 * (σ^2 + (μ)^2 - 1)

x = -10:0.1:10
y = 0.01:0.1:10
z = @. f(x', y)
plt1 = plot(x, y, z, xlims=[-10,10], ylims=[0,10], st=:surface, color=:plasma, size=[800,600], xlabel = L"\mu_1-\mu_2", ylabel = L"\sigma", zlabel = L"d_{KL}(P,Q)", title = L"KL divergence between $\mathcal{N}(\mu_1,\sigma)$ and $\mathcal{N}(\mu_2,1)$",camera=(60,10))
savefig(plt1,"ex2.png")