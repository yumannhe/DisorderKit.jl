using CairoMakie, Interpolations

function convolution_integral(f_vals, g_vals, x_vals)
    n = length(x_vals)
    dxs = diff(x_vals)
    C_vals = zeros(n)
    itp_g = LinearInterpolation(x_vals, g_vals, extrapolation_bc=0.0)

    for i in 1:n
        xi = x_vals[i]
        sum = 0.0
        for j in 1:i
            t = x_vals[j]
            dt = (j == 1) ? dxs[1] : dxs[j-1]  # handle variable spacing
            sum += f_vals[j] * itp_g(xi - t) * dt
        end
        C_vals[i] = sum
    end

    return C_vals
end

function dPdG(xs::Vector{Float64}, P::Vector{Float64}, R::Vector{Float64})
    dPdG = zeros(Float64, length(xs))
    dPdx = zeros(Float64, length(xs))
    for (i,x) in enumerate(xs)
        if i==1
            dx = xs[2] - xs[1]
            dP = (P[2] - P[1])
        elseif i == length(xs)
            dx = xs[end] - xs[end-1]
            dP = (P[end] - P[end-1])
        else
            dx = (xs[i+1] - xs[i-1])
            dP = (P[i+1] - P[i])
        end
        dPdx[i] = dP / dx
    end

    Cs = convolution_integral(P,P,xs)
    dPdG = dPdx .+ P.*(P[1]-R[1]) .+ R[1].*Cs
    return dPdG
end

function dRdG(xs::Vector{Float64}, P::Vector{Float64}, R::Vector{Float64})
    dRdG = zeros(Float64, length(xs))
    dRdx = zeros(Float64, length(xs))
    for (i,x) in enumerate(xs)
        if i==1
            dx = xs[2] - xs[1]
            dR = (R[2] - R[1])
        elseif i == length(xs)
            dx = xs[end] - xs[end-1]
            dR = (R[end] - R[end-1])
        else
            dx = (xs[i+1] - xs[i-1])
            dR = (R[i+1] - R[i])
        end
        dRdx[i] = dR / dx
    end

    Cs = convolution_integral(R,R,xs)
    dRdG = dRdX .+ R.*(R[1]-P[1]) .+ P[1].*Cs
    return dRdG
end

function log_uniform_distribution(m::Float64, d::Float64, xs::Vector{Float64}; Ω::Float64 = m+d)
    Ps = zeros(Float64,length(xs))
    for (i,x) in enumerate(xs)
        if x < log(Ω/(m+d))
            Ps[i] = 0.0
        elseif x > log(Ω/(m-d))
            Ps[i] = 0.0
        else
            Ps[i] = Ω/(2*d) * exp(-x)
        end
    end
    return Ps
end
dζ = 0.01
xs = [(0:dζ:2)...]
Ps0 = log_uniform_distribution(1., 0.3, xs; Ω = 1.3)
Rs0 = log_uniform_distribution(1., 0.3, xs; Ω = 1.3)

dΓ = 0.01
nsteps = 50

Ps = [Ps0]
Rs = [Rs0]
Cs = dPdG(xs, Ps[1], Rs[1])

for i in 1:nsteps
    dPdΓ = dPdG(xs, Ps[i], Rs[i])
    dRdΓ = dPdG(xs, Ps[i], Rs[i])
    Ps_new = max.(Ps[i] .+ dPdΓ * dΓ,0)
    Rs_new = max.(Rs[i] .+ dRdΓ * dΓ, 0)
    Ps_new = Ps_new ./ sum(Ps_new*dζ)  # Normalize
    Rs_new = Rs_new ./ sum(Rs_new*dζ)  # Normalize
    push!(Ps, Ps_new)
    push!(Rs, Rs_new)
end

set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$ζ$",
            ylabel = L"$P(ζ)$",
            # xscale = log10,
            # yscale = log10
            )
    ax2 = Axis(fig[1, 2], 
            xlabel = L"$β$",
            ylabel = L"$R(β)$",
            # xscale = log10,
            # yscale = log10
            )

           
plotsteps = 1:10:nsteps
for i in plotsteps
    scatter!(ax1, xs, Ps[i], label="Γ = $(i*dΓ)", markersize = 16)
    scatter!(ax2, xs, Rs[i], label="Γ = $(i*dΓ)", markersize = 16)
end
fig