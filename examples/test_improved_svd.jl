using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit, Crayons, LinearAlgebra, KrylovKit
using JLD2

# it seems that we still need the eigsolver, or the iteration would be too long, but since it's just for one 
# tensor but not contracted tensor, it might be easier to do 
# What we need to revise might just be some convergence condition etc.

function fixed_point_left(A, AL)
    # write in the disorderkit convention
    # the gauge would not change during this procedure
    function ftransfer(vl)
        @tensor vl[-1;-2] := vl[1;2]*A[2 3 4; 5 6 -2]*conj(AL[1 3 4;5 6 -1])
        return vl
    end
    return ftransfer   
end


function fixed_point_right(A, AR)
    # write in the disorderkit convention
    # the gauge would not change during this procedure
    function ftransfer(vr)
        @tensor vr[-1;-2] := vr[1;2]*A[-1 3 4; 5 6 1]*conj(AR[-2 3 4;5 6 2])
        return vr
    end
    return ftransfer   
end

# Define the gauge fixing SVD respectively
function iterate_lc(L, A; conv_tol=1e-8, max_iter=100, trunc=truncbelow(1e-8))
    @tensor A_mul[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
    U, S, Vt = tsvd(A_mul, ((1,2,3,4,5),(6,)); trunc=trunc)
    L_newp = permute(S*Vt, ((1,),(2,)))
    ulphase, L_new = leftorth(L_newp)
    U = U*ulphase
    L_new = L_new / norm(L_new)
    # # fix the gauge
    # @tensor Gaugei[-1;-2] := L_new[-1;1]*conj(L[-2;1]) 
    # Gaugei = diag(Gaugei)[Trivial()]
    # mag = abs.(Gaugei)
    # Gaugei = Gaugei ./mag
    # Gaugei = DiagonalTensorMap(Gaugei, L_new.space[1])
    # # @tensor L_new[-1;-2] := conj(Gaugei[-1;1])*L_new[1;-2]
    # L_new = adjoint(Gaugei)*L_new
    # U = U*Gaugei
    @tensor L_m[-1;-2] := conj(L[1;-1]) * L[1;-2]
    @tensor L_new_m[-1;-2] := conj(L_new[1;-1]) * L_new[1;-2]
    diff = norm(L_m-L_new_m)
    it = 0
    L = L_new
    L_list = [L, ]
    while (diff > conv_tol) && (it < max_iter)
        transferl = fixed_point_left(A, U)
        _, L_fixed = eigsolve(transferl, L, 1, :LM; tol = diff/10)
        # @show L_fixed
        # I think QR should be enough here. 
        # @show space(L_fixed[1])
        _, L = leftorth(L_fixed[1], ((1,),(2,)))
        L = L/norm(L)
        # L = L_fixed[1]/norm(L_fixed[1])
        @tensor L_m[-1;-2] := conj(L[1;-1]) * L[1;-2]
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
        U, S, Vt = tsvd(A_mul, ((1,2,3,4,5),(6,)); trunc=trunc)
        L_newp = permute(S*Vt, ((1,),(2,)))
        ulphase, L_new = leftorth(L_newp)
        U = U*ulphase
        L_new = L_new / norm(L_new)
        # # fix the gauge
        # @tensor Gaugei[-1;-2] := L_new[-1;1]*conj(L[-2;1]) 
        # Gaugei = diag(Gaugei)[Trivial()]
        # mag = abs.(Gaugei)
        # Gaugei = Gaugei ./mag
        # Gaugei = DiagonalTensorMap(Gaugei, L_new.space[1])
        # # @tensor L_new[-1;-2] := conj(Gaugei[-1;1])*L_new[1;-2]
        # L_new = adjoint(Gaugei)*L_new
        @tensor L_new_m[-1;-2] := conj(L_new[1;-1]) * L_new[1;-2]
        # U = U*Gaugei
        diff = norm(L_m-L_new_m)
        # @show it, diff
        it += 1
        L = L_new
        push!(L_list, L)
    end
    @show it
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return U,L_list,S
end
    

function iterate_rc(R, A; conv_tol=1e-8, max_iter=100, trunc=truncbelow(1e-8))
    @tensor A_mul[-1 -2 -3;-4 -5 -6] := A[-1 -2 -3;-4 -5 1] * R[1;-6]
    U, S, Vt = tsvd(A_mul, ((1,),(2,3,4,5,6)); trunc=trunc)
    R_newp = permute(U*S, ((1,),(2,)))
    R_new, vtrphase = rightorth(R_newp)
    Vt = vtrphase*Vt
    R_new = R_new / norm(R_new)
    # add a gauge fixer
    # @tensor Gaugei[-1;-2] := conj(R[1;-2])*R_new[1;-1]
    # Gaugei = diag(Gaugei)[Trivial()]
    # mag = abs.(Gaugei)
    # Gaugei = Gaugei ./mag
    # Gaugei = DiagonalTensorMap(Gaugei, R_new.space[1])
    # # @tensor R_new := R_new[-1;1]*conj(Gaugei[1;-2])
    # R_new = R_new*adjoint(Gaugei)
    # Vt = Gaugei * Vt
    @tensor R_m[-1;-2] := conj(R[-2;1]) * R[-1;1]
    @tensor R_new_m[-1;-2] := conj(R_new[-2;1]) * R_new[-1;1]
    diff = norm(R_m-R_new_m)
    it = 0
    R = R_new
    R_list = [R, ]
    while (diff > conv_tol) && (it < max_iter)
        transferr = fixed_point_right(A, Vt)
        _, R_fixed = eigsolve(transferr, R, 1, :LM; tol = diff/10)
        # I think QR should be enough here. 
        R,_ = rightorth(R_fixed[1], ((1,),(2,)))
        R = R/norm(R)
        @tensor R_m[-1;-2] := conj(R[-2;1]) * R[-1;1] 
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := A[-1 -2 -3;-4 -5 1] * R[1;-6]
        U, S, Vt = tsvd(A_mul, ((1,),(2,3,4,5,6)); trunc=trunc)
        R_newp = permute(U*S, ((1,),(2,)))
        R_new, urphase = rightorth(R_newp)
        Vt = urphase*Vt
        R_new = R_new / norm(R_new)
        # add a gauge fixer
        # @tensor Gaugei[-1;-2] := conj(R[1;-2])*R_new[1;-1]
        # Gaugei = diag(Gaugei)[Trivial()]
        # mag = abs.(Gaugei)
        # Gaugei = Gaugei ./mag
        # Gaugei = DiagonalTensorMap(Gaugei, R_new.space[1])
        # # @tensor R_new := R_new[-1;1]*conj(Gaugei[1;-2])
        # R_new = R_new*adjoint(Gaugei)
        # Vt = Gaugei * Vt
        @tensor R_new_m[-1;-2] := conj(R_new[-2;1]) * R_new[-1;1]
        diff = norm(R_m-R_new_m)
        it += 1
        R = R_new
        push!(R_list, R)
    end
    @show it
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return Vt,R_list,S
end


# test if this works and what would be the time cost
# load hte files
filename = "RTFIM_step_19.jld2"
ρlist, ρlist_truncated, Zinv_list = load(filename, "ρlist", "ρlist_truncated", "Zinv_list")

# Define model
N = 2
a = 0.7
b = 1.3

Js = Vector(a:(b-a)/(N-1):b)
hs = Vector(a:(b-a)/(N-1):b)
dτ = 5e-2
Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
Us = DisorderMPO([Us[1]])

# test the convergence and stability of the Us*Zinv alone
partitionL = ((1,2,3,4,5,),(6,))
partitionR = ((1,),(2,3,4,5,6,))
trunc = truncbelow(1e-8)
ALC_list = []
ARC_list = []
S_check_ρ = []
L_list = []
R_list = []
ite_keeper = length(ρlist)

R_list = []
ite_keeper = length(ρlist)
# ite_keeper = 1
for ite in 1:ite_keeper
    ρ = ρlist[ite][1]
    # initialize L and R, first check the unnormalized case
    Uρl, Sρl, Vtρl = tsvd(ρ, partitionL;trunc)
    Lρp = Sρl*Vtρl
    ulphase, Lρ = leftorth(Lρp)
    Uρl = Uρl*ulphase
    Lρ = Lρ / norm(Lρ)
    Uρr, Sρr, Vtρr = tsvd(ρ, partitionR;trunc)
    Rρp = Uρr*Sρr
    Rρ, urphase = rightorth(Rρp)
    Vtρr = urphase*Vtρr
    Rρ = Rρ / norm(Rρ)
    t_lc_svd = @elapsed begin
        Uρ, Lρ_list, S_Lρ = iterate_lc(Lρ, ρ;)
    end
    println("SVD Left took $(t_lc_svd) seconds") 
    t_rc_svd = @elapsed begin
        Vtρ, Rρ_list, S_Rρ = iterate_rc(Rρ, ρ;)
    end
    println("SVD Right took $(t_rc_svd) seconds") 
    # check the LR and entanglement spectrum 
    Lρfinal = Lρ_list[end]
    Rρfinal = Rρ_list[end]
    Cρ = Lρfinal*Rρfinal
    UL, Sρ, VtR = tsvd(Cρ; trunc=trunc)
    d = dim(space(Sρ,1))
    @show d, Sρ[40,40], Sρ[d,d]
    @tensor ALρ[-1 -2 -3; -4 -5 -6] := conj(UL[1 ;-1]) * Uρ[1 -2 -3;-4 -5 2] * UL[2;-6]
    @tensor ARρ[-1 -2 -3; -4 -5 -6] := VtR[-1 ;1] * Vtρ[1 -2 -3;-4 -5 2] * conj(VtR[-6;2])
    # @tensor ALρ[-1 -2 -3; -4 -5 -6] := Uρ[-1 -2 -3;-4 -5 2] * UL[2;-6]
    # @tensor ARρ[-1 -2 -3; -4 -5 -6] := VtR[-1 ;1] * Vtρ[1 -2 -3;-4 -5 -6]
    # # fidelity check:
    @tensor ALC[-1 -2 -3; -4 -5 -6] := ALρ[-1 -2 -3; -4 -5 1] * Sρ[1;-6]
    @tensor ARC[-1 -2 -3; -4 -5 -6] := Sρ[-1;1] * ARρ[1 -2 -3; -4 -5 -6]
    push!(S_check_ρ, Sρ)
    push!(ALC_list, ALC)
    push!(ARC_list, ARC)
    push!(L_list, Lρ_list)
    push!(R_list, Rρ_list)
    fidelity= norm(ALC-ARC)
    @info(crayon"blue"("Iteration $ite: Fidelity between left and right canonical forms = $fidelity"))
end






