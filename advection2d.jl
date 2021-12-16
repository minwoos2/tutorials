using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators, DomainSets, Distributions

u_exact = (x,t) -> exp.(-t) * cos.(x)
@parameters t x y
@variables u(..)
Dx = Differential(x)
Dy = Differential(y)
#Dxx = Differential(x)^2
#Dyy = Differential(y)^2
Dt = Differential(t)
t_min= 0.
t_max = 2.0
x_min = 0.
x_max = 2.
y_min = 0.
y_max = 2.
u1 = 10.0;
v1 = 10.0; 

nx_emis = 5
ny_emis = 5
nt_emis = 5
dx_emis = (x_max - x_min)/nx_emis
dy_emis = (y_max - y_min)/ny_emis
dt_emis = (t_max - t_min)/nt_emis


#Created random emission, later to be changed by real emission data.
emis = rand(LogNormal(1,3), nx_emis, ny_emis)

#Locate/ align emis into coordinate we're using
function emissions(x,y)
    i = Int(ceil((x - x_min)/dx_emis))
    j = Int(ceil((y - y_min)/dy_emis))
    #println(x, " ",y," ", i," ", j)
    return emis[i,j]
end

@register emissions(x, y)

eq  = Dt(u(t,x,y)) ~ u1*Dx(u(t,x,y)) + v1*Dy(u(t,x,y)) + emissions(x, y)

bcs = #[u(t_min,x,y) ~ analytic_sol_func(t_min,x,y),
      [u(t_min,x,y) ~ 0,
       u(t,x_min,y) ~ 0,
       u(t,x_max,y) ~ 0,
       u(t,x,y_min) ~ 0,
       u(t,x,y_max) ~ 0]

# Space and time domains
domains = [t ∈ IntervalDomain(t_min,t_max),
           x ∈ IntervalDomain(x_min,x_max),
           y ∈ IntervalDomain(y_min,y_max)]
pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])

# Method of lines discretization
dx = 0.1; dy = 0.1
discretization = MOLFiniteDifference([x=>dx,y=>dy],t)
prob = ModelingToolkit.discretize(pdesys,discretization)
sol = solve(prob,saveat = 0.1, Tsit5())

using Plots
xs,ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]

u_sol = reshape(sol.u[1], length(xs)-2,length(ys)-2)
p1 = plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")
u_sol = reshape(sol.u[5], length(xs)-2,length(ys)-2)
p2 = plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")
u_sol = reshape(sol.u[10], length(xs)-2,length(ys)-2)
p3 = plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")
u_sol = reshape(sol.u[15], length(xs)-2,length(ys)-2)
p4 = plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")
plottogether = plot(p1,p2,p3,p4)

anim = @animate for i ∈ 1:length(sol.t)
    u_sol = reshape(sol.u[i], length(xs)-2,length(ys)-2)
    plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")


end
gif(anim, "anim_fps15.gif", fps = 15)
