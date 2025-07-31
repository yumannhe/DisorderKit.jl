using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using DisorderKit, TimerOutputs, CairoMakie, LsqFit
# Define model
N = 2
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
trunctol = 1e-6
D_max = 20
D_Z = 2
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 50, verbosity = 2)
alg_trunc_Z = StandardTruncation(trunc_method = truncerr(trunctol))


# alg_trunc_disordermpo = DisorderTracedTruncation(trunc_method = truncdim(D_max))
alg_trunc_disordermpo = DisorderOpenTruncation(trunc_method = truncdim(D_max))


βs = 1:0.5:5
# Evolve density matrix
function get_ξ(βs, Us, ps, alg_trunc_Z)
    ξs = zeros(length(βs))
    DZs = zeros(Int,length(βs))
    ents = Vector{Float64}[]
    # ents = Float64[]
    ϵs = []
    ϵs2 = Float64[]
    ρ0 = nothing
    nsteps = round(Int, βs[1]/dτ)
    for (i,β) in enumerate(βs)
        @show (i,β)
        if β > βs[1]
            dβ = βs[i] - βs[i-1]
            nsteps = round(Int, dβ/dτ)
        end
        inversion_frequency = 1
        alg_evolution = iDTEBD(alg_inversion, alg_trunc_Z, alg_trunc_disordermpo; invtol = invtol, nsteps = nsteps, verbosity = 2, truncfrequency = 1, inversion_frequency = inversion_frequency, timer_output = TimerOutput(), max_inverse_dim = 4)
        ρs, ϵ = evolve_densitymatrix(Us, ps, alg_evolution; ρ0 = ρ0)
        ξs[i] = average_correlation_length(ρs, ps)
        Z = partition_functions(ρs)
        # Z = truncate_mpo(Z, alg_trunc_Z)
        DZs[i] = dim(space(Z[1])[1])
        ϵ2 = DisorderKit.test_identity(Z)
        push!(ϵs2, ϵ2)
        entval = DisorderKit.entanglement_spectrum(Z, i)
        push!(ents, entval)
        # _, S, _ = tsvd(Z[1], (1,2,3), (4,))
        # entval = S.data./S.data[1]
        ρ0 = ρs
        push!(ϵs, ϵ...)
    end
    return ξs, ϵs, DZs, ents, ϵs2
end

ξs, ϵs, DZs, ents, ϵs2 = get_ξ(βs, Us, ps, alg_trunc_Z)
ents = reduce(hcat, ents)

# Set up the figure and axes
set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$i$",
            ylabel = L"$s_i$",
            # xscale = log10,
            yscale = log10
            )
    ax2 = Axis(fig[2, 1], 
            xlabel = L"$\ln β$",
            ylabel = L"$s_i$",
    #         # xscale = log10,
            yscale = log10
            )

    ax3 = Axis(fig[3, 1], 
        xlabel = L"$\ln β$",
        ylabel = L"$ϵ$",
    #         # xscale = log10,
        yscale = log10
        )


number_eigs = 10
plotbs = 1:2:length(βs)

for i in plotbs
    label = "β = $(βs[i])"
    scatter!(ax1, 1:number_eigs, ents[1:number_eigs, i], label=label,markersize = 16)
end

for i in 1:8
    label = "i = $(i)"
    scatter!(ax2, log.(βs), ents[i,:], label=label, markersize = 16)
    lines!(ax2, log.(βs), ones(length(βs))*1e-6, linewidth = 1)
end
scatter!(ax3,dτ:dτ:βs[end],ϵs.+1e-16,markersize = 16)
lines!(ax3, dτ:dτ:βs[end], ones(length(ϵs))*1e-6, linewidth = 1)
scatter!(ax3, βs, ϵs2, markersize = 16)


fig[1, 2] = Legend(fig, ax1, framevisible = false)
fig[2, 2] = Legend(fig, ax2, framevisible = false)
fig

save("entanglement_open4_invtol1e-8.png",fig)