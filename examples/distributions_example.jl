using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using DisorderKit, TimerOutputs, CairoMakie, LsqFit

h0 = [0.7, 1.0, 1.3]
h1 = [0.3, 0.8, 1.3]
h2 = [0.1, 0.7, 1.3]
dists = [h1]

dτ = 5e-2

# Define algorithms
invtol = 1e-6
D_max = 20
D_z = 2
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 50, verbosity = 2)
alg_trunc_Z = StandardTruncation(trunc_method = truncdim(D_z))
alg_trunc_disordermpo = DisorderOpenTruncation(trunc_method = truncdim(D_max))


βs = 1:0.5:10
# Evolve density matrix
function get_ξ(βs, Us, ps, alg_trunc_disordermpo)
    ξs = zeros(length(βs))
    DZs = zeros(Int,length(βs))
    times = zeros(Float64,length(βs))
    ϵs = []
    ρ0 = nothing
    nsteps = round(Int, βs[1]/dτ)
    for (i,β) in enumerate(βs)
        @show (i,β)
        if β > βs[1]
            dβ = βs[i] - βs[i-1]
            nsteps = round(Int, dβ/dτ)
        end
        inversion_frequency = 1
        alg_evolution = iDTEBD(alg_inversion, alg_trunc_Z, alg_trunc_disordermpo; invtol = invtol, nsteps = nsteps, verbosity = 2, truncfrequency = 1, inversion_frequency = inversion_frequency, timer_output = TimerOutput(), max_inverse_dim = 2)
        ρs, ϵ = evolve_densitymatrix(Us, ps, alg_evolution; ρ0 = ρ0)
        ξs[i] = average_correlation_length(ρs, ps)
        time_elapsed = TimerOutputs.time(alg_evolution.timer_output["truncate_disorder_MPO"])
        times[i] = 1e-9*time_elapsed./ TimerOutputs.ncalls(alg_evolution.timer_output["truncate_disorder_MPO"])
        ρ0 = ρs
        push!(ϵs, ϵ...)
    end
    return ξs, ϵs, times
end

# Set up the figure and axes
set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$(\ln{β})^2$",
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
            )
    ax2 = Axis(fig[1, 2], 
            xlabel = L"$β$",
            ylabel = L"$ϵ$",
    #         # xscale = log10,
            yscale = log10
            )
    ax3 = Axis(fig[2, 1], 
            xlabel = L"$β$",
            ylabel = L"$\bar{t}_{\text{trunc}}(s)$",
            # xscale = log10,
            # yscale = log10
    )
    ax4 = Axis(fig[2, 2], 
        xlabel = L"$β$",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
    )


for (ix, d) in enumerate(dists)
    # Define model
    hs = d
    Js = hs
    N = length(d)
    ps = ones(N^2)./N^2

    # Define Time-Evolution Operator
    Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
    UsTrotter = DisorderMPO([Us[1]])

    ξs, ϵs, times = get_ξ(βs, Us, ps, alg_trunc_disordermpo)

    # Rescale ξs
    # ξs ./= ξs[1]
    # Make fits
    minfit = 5
    maxfit = length(ξs)
    linmodel(t, p) = p[1] .+ p[2]*t
    p0 = [1., 1.]
    linfit = curve_fit(linmodel, log.(βs[1:maxfit]).^2, ξs[1:maxfit], p0)
    global linparams = linfit.param

    # Plot correlation lengths in function of β
    scatter!(ax1,log.(βs).^2, ξs, label=L"$d=%$(d)$",markersize = 16)
    lines!(ax1,log.(βs).^2, linmodel(log.(βs).^2,linparams), label=L"$p_2=%$(linparams[2])$")

    scatter!(ax2,dτ:dτ:βs[end],ϵs.+1e-16, label=L"$d=%$(d)$",markersize = 16)

    scatter!(ax3, βs, times, label=L"$d=%$(d)$",markersize = 16)

    scatter!(ax4,βs,ξs, label=L"$d=%$(d)$",markersize = 16)
end

fig[1, 3] = Legend(fig, ax1, framevisible = false)
fig

# save("taylor_open.png",fig)
