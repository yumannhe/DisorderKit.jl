using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit, Crayons, LinearAlgebra
using JLD2

# load hte files
filename = "RTFIM_step_18.jld2"
ρlist, ρlist_truncated, Zinv_list = load(filename, "ρlist", "ρlist_truncated", "Zinv_list")

# define the iteration function
conv_tol = 1e-8
max_iter = 100
function iterate_lc(L, A; conv_tol=conv_tol, max_iter=max_iter)
    diff = 1
    it = 0
    U = nothing
    space_o = L.space[1]
    S_old = isomorphism(ComplexF64, space_o, space_o)
    L_list = [L,]
    @tensor L_m[-1;-2] := conj(L[1;-1]) * L[1;-2]
    # L_ite = deepcopy(L)
    # it would be better to have a gauge fixer
    while (diff > conv_tol) && (it < max_iter)
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
        U, S, Vt = tsvd(A_mul, ((1,2,3,4,5),(6,)); trunc=trunc)
        L_new = permute(S*Vt, ((1,),(2,)))
        L_new = L_new / norm(L_new)
        # add a gauge fixer
        @tensor Gaugei[-1;-2] := L_new[-1;1]*conj(L[-2;1]) 
        Gaugei = diag(Gaugei)[Trivial()]
        mag = abs.(Gaugei)
        Gaugei = Gaugei ./mag
        Gaugei = DiagonalTensorMap(Gaugei, L_new.space[1])
        # @tensor L_new[-1;-2] := conj(Gaugei[-1;1])*L_new[1;-2]
        L_new = adjoint(Gaugei)*L_new
        U = U*Gaugei
        @tensor L_new_m[-1;-2] := conj(L_new[1;-1]) * L_new[1;-2]
        # U = permute(U*sqrt(S), ((1,2),(3,)))
        # normalize L_new
        # diff = norm(S_old - S)
        diff = norm(L_m - L_new_m)
        L_m = L_new_m
        @show diff, it
        L = L_new
        L_list = push!(L_list, L)
        S_old = S
        it += 1
    end
    @show it
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return U,L_list,S_old
end


function iterate_rc(R, A; conv_tol=conv_tol, max_iter=max_iter)
    diff = 1
    it = 0
    Vt = nothing
    space_o = R.space[2]
    S_old = isomorphism(ComplexF64, space_o', space_o')
    R_list = [R,]
    @tensor R_m[-1;-2] := R[-1;1] * conj(R[-2;1])
    while (diff > conv_tol) && (it < max_iter)
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := A[-1 -2 -3;-4 -5 1] * R[1;-6]
        U, S, Vt = tsvd(A_mul, ((1,),(2,3,4,5,6)); trunc=trunc)
        R_new = permute(U*S, ((1,),(2,)))
        R_new = R_new / norm(R_new)
        # add a gauge fixer
        @tensor Gaugei[-1;-2] := conj(R[1;-2])*R_new[1;-1]
        Gaugei = diag(Gaugei)[Trivial()]
        mag = abs.(Gaugei)
        Gaugei = Gaugei ./mag
        Gaugei = DiagonalTensorMap(Gaugei, R_new.space[1])
        # @tensor R_new := R_new[-1;1]*conj(Gaugei[1;-2])
        R_new = R_new*adjoint(Gaugei)
        Vt = Gaugei * Vt
        @tensor R_new_m[-1;-2] := R_new[-1;1] * conj(R_new[-2;1])
        # U = permute(U*sqrt(S), ((1,2),(3,)))
        # normalize L_new
        # diff = norm(S_old - S)
        diff = norm(R_m - R_new_m)
        R_m = R_new_m
        @show diff, it
        R = R_new
        R_list = push!(R_list, R)
        S_old = S
        it += 1
    end
    @show it
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return Vt,R_list,S_old
end

function iterate_lc_qr(L, A; conv_tol=conv_tol, max_iter=max_iter)
    diff = 1
    it = 0
    U = nothing
    L_list = [L,]
    # @tensor L_m[-1;-2] := conj(L[1;-1]) * L[1;-2]
    # L_ite = deepcopy(L)
    # it would be better to have a gauge fixer
    while (diff > conv_tol) && (it < max_iter)
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
        U, L_new = leftorth(A_mul, ((1,2,3,4,5),(6,));)
        L_new = L_new / norm(L_new)
        # add a gauge fixer
        # @tensor Gaugei[-1;-2] := L_new[-1;1]*conj(L[-2;1]) 
        # Gaugei = diag(Gaugei)[Trivial()]
        # mag = abs.(Gaugei)
        # Gaugei = Gaugei ./mag
        # Gaugei = DiagonalTensorMap(Gaugei, L_new.space[1])
        # # @tensor L_new[-1;-2] := conj(Gaugei[-1;1])*L_new[1;-2]
        # L_new = adjoint(Gaugei)*L_new
        # U = U*Gaugei
        # @tensor L_new_m[-1;-2] := conj(L_new[1;-1]) * L_new[1;-2]
        # U = permute(U*sqrt(S), ((1,2),(3,)))
        # normalize L_new
        # diff = norm(S_old - S)
        # diff = norm(L_m - L_new_m)
        diff = norm(L - L_new)
        # L_m = L_new_m
        @show diff, it
        L = L_new
        L_list = push!(L_list, L)
        it += 1
    end
    @show it
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return U,L_list
end


function iterate_rc_qr(R, A; conv_tol=conv_tol, max_iter=max_iter)
    diff = 1
    it = 0
    Vt = nothing
    R_list = [R,]
    # @tensor R_m[-1;-2] := R[-1;1] * conj(R[-2;1])
    while (diff > conv_tol) && (it < max_iter)
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := A[-1 -2 -3;-4 -5 1] * R[1;-6]
        R_new, Vt = rightorth(A_mul, ((1,),(2,3,4,5,6));)
        # R_new = permute(U*S, ((1,),(2,)))
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
        # @tensor R_new_m[-1;-2] := R_new[-1;1] * conj(R_new[-2;1])
        # U = permute(U*sqrt(S), ((1,2),(3,)))
        # normalize L_new
        # diff = norm(S_old - S)
        # diff = norm(R_m - R_new_m)
        diff = norm(R - R_new)
        # R_m = R_new_m
        @show diff, it
        R = R_new
        R_list = push!(R_list, R)
        it += 1
    end
    @show it
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return Vt,R_list
end

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
trunc = truncbelow(1e-12)
ALC_list = []
ARC_list = []
S_check_ρ = []
L_list = []
R_list = []
# ite_keeper = length(ρlist)
ite_keeper = 1
for ite in 1:ite_keeper
    ρ = ρlist[ite][1]
    ρ_truncated = ρlist_truncated[ite]
    Zinv = Zinv_list[ite]
    # initialize L and R, first check the unnormalized case
    Ql, Lqrρ = leftorth(ρ, partitionL)
    Lqrρ = Lqrρ/norm(Lqrρ)
    Rqrρ, Qr = rightorth(ρ, partitionR)
    Rqrρ = Rqrρ/norm(Rqrρ)
    Uρl, Sρl, Vtρl = tsvd(ρ, partitionL;trunc)
    Lρ = Sρl*Vtρl
    Lρ = Lρ / norm(Lρ)
    Uρr, Sρr, Vtρr = tsvd(ρ, partitionR;trunc)
    Rρ = Uρr*Sρr
    Rρ = Rρ / norm(Rρ)
    t_lc_svd = @elapsed begin
        Uρ, Lρ_list, S_Lρ = iterate_lc(Lρ, ρ;)
    end
    println("SVD Left took $(t_lc_svd) seconds") 
    t_rc_svd = @elapsed begin
        Vtρ, Rρ_list, S_Rρ = iterate_rc(Rρ, ρ;)
    end
    println("SVD Right took $(t_rc_svd) seconds") 
    t_lc_qr = @elapsed begin
        Uqr, Lqr = iterate_lc_qr(Lqrρ, ρ)
    end
    println("QR Left took $(t_lc_qr) seconds") 
    t_rc_qr = @elapsed begin
        Vtqr, Rqr  = iterate_rc_qr(Rqrρ, ρ)
    end
    println("QR Right took $(t_rc_qr) seconds") 
    # check the LR and entanglement spectrum 
    Lρfinal = Lρ_list[end]
    Rρfinal = Rρ_list[end]
    Cρ = Lρfinal*Rρfinal
    UL, Sρ, VtR = tsvd(Cρ; trunc=trunc)
    d = dim(space(Sρ,1))
    @show d, Sρ[40,40], Sρ[d,d]
    # check the LR and entanglement spectrum for qr results
    Lρqrfinal = Lqr[end]
    Rρqrfinal = Rqr[end]
    Cρqr = Lρqrfinal*Rρqrfinal
    ULqr, Sρqr, VtRqr = tsvd(Cρqr; trunc=trunc)
    @show Sρqr[40,40], Sρqr[d,d]
    @show norm(Sρ-Sρqr)
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

#-----------------check the inversion step entanglement stucture-------------
# record the final entanglement structure and record the truncation time
D_max = 40
ps = ones(N^2)./N^2
S_check_Tinv = []
alg_trunc_disordermpo = DisorderOpenTruncation(trunc_method = truncdim(D_max))
for ite in 1:ite_keeper
    ρ = ρlist[ite]
    Zinv = Zinv_list[ite][1]
    @tensor Tinv[-1 -2 -3 -4;-5 -6 -7 -8] := Us[1][-1 -3 1;-5 -6 -7]*Zinv[-2 -4;1 -8]
    iso_v = isomorphism(fuse(space(Tinv,1),space(Tinv,2)), space(Tinv,1)⊗space(Tinv,2))
    @tensor Tinv[-1 -2 -3;-4 -5 -6] := iso_v[-1;1 2]*Tinv[1 2 -2 -3;-4 -5 3 4]*conj(iso_v[-6;3 4])
    # initialize L and R, first check the unnormalized case
    Uρl, Sρl, Vtρl = tsvd(Tinv, partitionL;trunc)
    Lρ = Sρl*Vtρl
    Lρ = Lρ / norm(Lρ)
    Uρr, Sρr, Vtρr = tsvd(Tinv, partitionR;trunc)
    Rρ = Uρr*Sρr
    Rρ = Rρ / norm(Rρ)
    t_lc_svd = @elapsed begin
        Uρ, Lρ_list, S_Lρ = iterate_lc(Lρ, Tinv;)
    end
    println("SVD Left took $(t_lc_svd) seconds") 
    t_rc_svd = @elapsed begin
        Vtρ, Rρ_list, S_Rρ = iterate_rc(Rρ, Tinv;)
    end
    println("SVD Right took $(t_rc_svd) seconds") 
    t_truncation = @elapsed begin
        ρs_truncated = truncate_mpo(ρ, ps, alg_trunc_disordermpo)
    end
    println("The truncation took $(t_truncation) seconds") 
    # check the LR and entanglement spectrum 
    Lρfinal = Lρ_list[end]
    Rρfinal = Rρ_list[end]
    Cρ = Lρfinal*Rρfinal
    UL, Sρ, VtR = tsvd(Cρ; trunc=trunc)
    @show Sρ
    push!(S_check_Tinv, Sρ)
    @info(crayon"blue"("Iteration $ite done."))
end



