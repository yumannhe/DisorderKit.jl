using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit

# Define model
N = 2
a = 0.7
b = 1.3

Hs = 0:0.005:0.02
Js = Vector(a:(b-a)/(N-1):b)
hs = Vector(a:(b-a)/(N-1):b)
ps = ones(N^2)./N^2
# Js = [1.0]
# hs = [1.0]
# ps = [1.0]

# Define Time-Evolution Operator
dτ = 5e-2

# Define algorithms
invtol = 1e-6
trunctol = 1e-6
D_max = 20
D_z = 4
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)
alg_trunc_Z = StandardTruncation(trunc_method = truncdim(D_z))
alg_trunc_disordermpo = DisorderOpenTruncation(trunc_method = truncdim(D_max))

βs = 1:0.5:5
# Evolve density matrix
function get_ξ(βs, Us)
    Z = TensorMap(zeros(ComplexF64, 2, 2),ℂ^2,ℂ^2)
    Z[1, 1], Z[2, 2] = 1, -1
    ξs = zeros(length(βs))
    Ms = zeros(ComplexF64,length(βs))
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
        Ms[i] = measure(ρs, ps, Z, 1)
        ρ0 = ρs
        push!(ϵs, ϵ...)
    end
    return ξs, Ms, ϵs
end

Msflat = []
ξsflat = []
for H in Hs
    Us = RTFIM_time_evolution_Trotter(dτ, hs, Js, H)
    Us = DisorderMPO([Us[1]])
    ξ, M, ϵs = get_ξ(βs, Us)
    push!(Msflat,M)
    push!(ξsflat, ξ)
end
Msred = real.(reduce(hcat, Msflat))
ξsred = real.(reduce(hcat, ξsflat))

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
            xlabel = L"$H$",
            ylabel = L"$<σ^z>$",
            # xscale = log10,
            # yscale = log10
    )
    ax4 = Axis(fig[2, 2], 
        xlabel = L"$(\ln{H})^2$",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
    )

for (ih, h) in enumerate(Hs)
    # Plot correlation lengths in function of β
    scatter!(ax1,log.(βs).^2, ξsred[:,ih], label=L"H=%$(h)",markersize = 16)
end

βsplotind = 1:4:12
for ib in βsplotind
    # # Plot inversion accuracy in function of β
    # scatter!(ax2,dτ:dτ:βs[end],ϵs.+1e-16, label=L"D=%$(D_max)",markersize = 16)

    # Plot average thermal energy in function of β
    scatter!(ax3, log.(Hs[2:end]), Msred[ib,:][2:end], label=L"$β = %$(βs[ib])$",markersize = 16)
    # Plot specific heat in function of β
    scatter!(ax4, log.(Hs[2:end]).^2, ξsred[ib,:][2:end], label=L"$β = %$(βs[ib])$",markersize = 16)

end

fig[1, 3] = Legend(fig, ax1, framevisible = false)
fig[2, 3] = Legend(fig, ax3, framevisible = false)
fig
