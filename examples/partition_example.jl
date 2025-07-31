using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using DisorderKit, TimerOutputs, CairoMakie, LsqFit
# Define model
N = 3
a = 0.7
b = 1.3
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2

# Define Time-Evolution Operator
dτ = 5e-2
Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
Us = DisorderMPO([Us[1]])


# Define algorithms
invtol = 1e-8
trunctol = 1e-8
D_max = 20
D_Z = 2
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 50, verbosity = 2)
alg_trunc_Z1 = StandardTruncation(trunc_method = truncerr(trunctol))
alg_trunc_Z2 = StandardTruncation(trunc_method = truncdim(D_Z))


# alg_trunc_disordermpo = DisorderTracedTruncation(trunc_method = truncdim(D_max))
alg_trunc_disordermpo = DisorderOpenTruncation(trunc_method = truncdim(D_max))

truncation_algorithms = [alg_trunc_Z1, alg_trunc_Z2]
labels = [L"\text{Truncerr}", L"\text{Truncdim}"]


βs = 1:0.5:15
# Evolve density matrix
function get_ξ(βs, Us, ps, alg_trunc_Z)
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
        Z = partition_functions(ρs)
        Z = truncate_mpo(Z, alg_trunc_Z)
        DZs[i] = dim(space(Z[1])[1])
        time_elapsed = TimerOutputs.time(alg_evolution.timer_output["truncate_disorder_MPO"])
        times[i] = 1e-9*time_elapsed./ TimerOutputs.ncalls(alg_evolution.timer_output["truncate_disorder_MPO"])
        ρ0 = ρs
        push!(ϵs, ϵ...)
    end
    return ξs, ϵs, DZs, times
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
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
    )
    ax4 = Axis(fig[2, 2], 
        xlabel = L"$β$",
        ylabel = L"$D_\mathcal{Z}$",
        # xscale = log10,
        # yscale = log10
    )


for (ix, alg) in enumerate(truncation_algorithms)

    ξs, ϵs, DZs, times = get_ξ(βs, Us, ps, alg)

    # Make fits
    minfit = 4
    maxfit = length(ξs)
    linmodel(t, p) = p[1] .+ p[2]*t
    p0 = [1., 1.]
    linfit = curve_fit(linmodel, log.(βs[minfit:maxfit]).^2, ξs[minfit:maxfit], p0)
    global linparams = linfit.param
    quadmodel(t, p) = p[1] .+ p[2]*t .+ p[3]*t.^2
    q0 = [1., 1., 1.]
    quadfit = curve_fit(quadmodel, log.(βs[minfit:maxfit]), ξs[minfit:maxfit], q0)
    global quadparams = quadfit.param

    # Plot correlation lengths in function of β
    scatter!(ax1,log.(βs).^2, ξs, label=labels[ix],markersize = 16)
    lines!(ax1,log.(βs).^2, linmodel(log.(βs).^2,linparams), label=L"$p_2=%$(linparams[2])$")
    lines!(ax1,log.(βs).^2, quadmodel(log.(βs),quadparams), label=L"$p_3=%$(quadparams[3])$")


    scatter!(ax2,dτ:dτ:βs[end],ϵs.+1e-16, label=labels[ix],markersize = 16)

    scatter!(ax3, βs, ξs, label=labels[ix],markersize = 16)
    lines!(ax3, βs, linmodel(log.(βs).^2,linparams), label=L"$p_2=%$(linparams[2])$")
    lines!(ax3, βs, quadmodel(log.(βs),quadparams), label=L"$p_3=%$(quadparams[3])$")

    scatter!(ax4,βs,DZs, label=labels[ix],markersize = 16)
end

fig[1, 3] = Legend(fig, ax1, framevisible = false)
fig

# save("truncZ_open5.png",fig)
