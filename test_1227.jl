using Plots

x = 0:0.01:2pi
f(x) = sin(x)
plot(x, f)

g(x) = cos(x)
plot!(x, g)