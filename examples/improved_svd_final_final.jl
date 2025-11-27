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


function leftorth_trunc(L,A;trunc=trunc, do_qr=false)
    @tensor A_mul[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
    U, S, Vt = tsvd(A_mul, ((1,2,3,4,5),(6,)); trunc=trunc)
    L_newp = permute(S*Vt, ((1,),(2,)))
    ulphase, L_new = leftorth(L_newp)
    U = permute(U*ulphase, ((1,2,3,),(4,5,6,)))
    L_new = L_new / norm(L_new)
    @tensor L_m[-1;-2] := conj(L[1;-1]) * L[1;-2]
    @tensor L_new_m[-1;-2] := conj(L_new[1;-1]) * L_new[1;-2]
    diff = norm(L_m - L_new_m)
    return U, L_new, diff,S
end


function rightorth_trunc(R,A;trunc=trunc)
    @tensor A_mul[-1 -2 -3;-4 -5 -6] := A[-1 -2 -3;-4 -5 1] * R[1;-6]
    U, S, Vt = tsvd(A_mul, ((1,),(2,3,4,5,6)); trunc=trunc)
    R_newp = permute(U*S, ((1,),(2,)))
    R_new, vtrphase = rightorth(R_newp)
    Vt = vtrphase*Vt
    R_new = R_new / norm(R_new)
    @tensor R_m[-1;-2] := conj(R[-2;1]) * R[-1;1]
    @tensor R_new_m[-1;-2] := conj(R_new[-2;1]) * R_new[-1;1]
    diff = norm(R_m-R_new_m)
    return Vt, R_new, diff, S
end


# Define the gauge fixing SVD respectively
function iterate_lc(L, A; conv_tol=1e-12, max_iter=200, trunc=truncbelow(1e-10))
    it = 0
    U, L_new, diff, S = leftorth_trunc(L,A;trunc)
    L = L_new
    # check if the space match
    dimL, dim_U = space(L,1), space(U,1)
    while dimL != dim_U
        U, L_new, diff, S = leftorth_trunc(L,A;trunc)
        L=L_new
        dimL, dim_U = space(L,1), space(U,1)
        it +=1
    end
    while (diff > conv_tol) && (it < max_iter)
        @show space(L), space(A), space(U)
        # @tensor LA[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
        transferl = fixed_point_left(A, U)
        # transferl = fixed_point_left(A, LA)
        _, L_fixed = eigsolve(transferl, L, 1, :LM; tol = diff/10)
        _, L = leftorth(L_fixed[1], ((1,),(2,)))
        L = L/norm(L)
        U, L_new, diff, S = leftorth_trunc(L,A;trunc)
        @show space(L_new),diff
        L = L_new
        # check if the space match
        dimL, dim_U = space(L,1), space(U,1)
        it +=1
        while dimL != dim_U
            U, L_new, diff, S = leftorth_trunc(L,A;trunc)
            L=L_new
            dimL, dim_U = space(L,1), space(U,1)
            it +=1
        end
    end
    @show it, space(S,1)
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return U,L,S
end
    

function iterate_rc(R, A; conv_tol=1e-12, max_iter=200, trunc=truncbelow(1e-10))
    it = 0
    Vt, R_new, diff, S = rightorth_trunc(R,A;trunc)
    R=R_new
    #check the dimension
    dimR, dim_Vt = space(R,2), space(Vt,6)
    while dimR != dim_Vt
        Vt, R_new, diff, S = rightorth_trunc(R,A;trunc)
        R=R_new
        #check the dimension
        dimR, dim_Vt = space(R,2), space(Vt,6)
        it +=1
    end
    while (diff > conv_tol) && (it < max_iter)
        @show space(R), space(A), space(Vt)
        transferr = fixed_point_right(A, Vt)
        _, R_fixed = eigsolve(transferr, R, 1, :LM; tol = diff/10)
        # I think QR should be enough here. 
        R,_ = rightorth(R_fixed[1], ((1,),(2,)))
        R = R/norm(R)
        Vt, R_new, diff, S = rightorth_trunc(R, A;trunc)
        @show space(R_new),diff
        R = R_new
        dimR, dim_Vt = space(R,2), space(Vt,6)
        it +=1
        while dimR != dim_Vt    
            Vt, R_new, diff, S = rightorth_trunc(R,A;trunc)
            R=R_new
            #check the dimension
            dimR, dim_Vt = space(R,2), space(Vt,6)
            it +=1
        end
    end
    @show it, space(S,1)
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return Vt,R,S
end


# test if this works and what would be the time cost
# load hte files
filename = "RTFIM_step_1.jld2"
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
D_max = 40

# test the convergence and stability of the Us*Zinv alone
partitionL = ((1,2,3,4,5,),(6,))
partitionR = ((1,),(2,3,4,5,6,))
trunc = truncbelow(1e-10)
ALC_list = []
ARC_list = []
S_check_ρ = []
ite_keeper = length(ρlist)
truncfinal = truncdim(D_max)
R_list = []
# ite_keeper = length(ρlist)
ite_keeper = 1
for ite in 1:ite_keeper
    ρ = ρlist[ite][1]
    # initialize L and R, first check the unnormalized case
    @show space(ρ)
    Uρl, Sρl, Vtρl = tsvd(ρ, partitionL;trunc)
    @show space(Sρl)
    Lρp = Sρl*Vtρl
    ulphase, Lρ = leftorth(Lρp)
    Uρl = Uρl*ulphase
    Lρ = Lρ / norm(Lρ)
    Uρr, Sρr, Vtρr = tsvd(ρ, partitionR;trunc)
    @show space(Sρr)
    Rρp = Uρr*Sρr
    Rρ, urphase = rightorth(Rρp)
    Vtρr = urphase*Vtρr
    Rρ = Rρ / norm(Rρ)
    t_lc_svd = @elapsed begin
        Uρ, Lρfinal, S_Lρ = iterate_lc(Lρ, ρ;)
    end
    println("SVD Left took $(t_lc_svd) seconds") 
    t_rc_svd = @elapsed begin
        Vtρ, Rρfinal, S_Rρ = iterate_rc(Rρ, ρ;)
    end
    println("SVD Right took $(t_rc_svd) seconds") 
    # check the LR and entanglement spectrum 
    Cρ = Lρfinal*Rρfinal
    # UL, Sρ, VtR = tsvd(Cρ; trunc=truncfinal)
    UL, Sρ, VtR = tsvd(Cρ; trunc=trunc)
    d = dim(space(Sρ,1))
    @show d, Sρ[d,d]
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
    fidelity= norm(ALC-ARC)
    @info(crayon"blue"("Iteration $ite: Fidelity between left and right canonical forms = $fidelity"))
end


#-----------------check the inversion step entanglement stucture-------------
# record the final entanglement structure and record the truncation time
# ps = ones(N^2)./N^2
# S_check_Tinv = []
# alg_trunc_disordermpo = DisorderOpenTruncation(trunc_method = truncdim(D_max))
# for ite in 1:ite_keeper
#     ρ = ρlist[ite]
#     Zinv = Zinv_list[ite][1]
#     @tensor Tinv[-1 -2 -3 -4;-5 -6 -7 -8] := Us[1][-1 -3 1;-5 -6 -7]*Zinv[-2 -4;1 -8]
#     iso_v = isomorphism(fuse(space(Tinv,1),space(Tinv,2)), space(Tinv,1)⊗space(Tinv,2))
#     @tensor Tinv[-1 -2 -3;-4 -5 -6] := iso_v[-1;1 2]*Tinv[1 2 -2 -3;-4 -5 3 4]*conj(iso_v[-6;3 4])
#     # initialize L and R, first check the unnormalized case
#     Uρl, Sρl, Vtρl = tsvd(Tinv, partitionL;trunc)
#     Lρ = Sρl*Vtρl
#     Lρ = Lρ / norm(Lρ)
#     Uρr, Sρr, Vtρr = tsvd(Tinv, partitionR;trunc)
#     Rρ = Uρr*Sρr
#     Rρ = Rρ / norm(Rρ)
#     t_lc_svd = @elapsed begin
#         Uρ, Lρfinal, S_Lρ = iterate_lc(Lρ, Tinv;)
#     end
#     println("SVD Left took $(t_lc_svd) seconds") 
#     t_rc_svd = @elapsed begin
#         Vtρ, Rρfinal, S_Rρ = iterate_rc(Rρ, Tinv;)
#     end
#     println("SVD Right took $(t_rc_svd) seconds") 
#     t_truncation = @elapsed begin
#         ρs_truncated = truncate_mpo(ρ, ps, alg_trunc_disordermpo)
#     end
#     println("The truncation took $(t_truncation) seconds") 
#     # check the LR and entanglement spectrum 
#     Cρ = Lρfinal*Rρfinal
#     UL, Sρ, VtR = tsvd(Cρ; trunc=trunc)
#     @show Sρ
#     push!(S_check_Tinv, Sρ)
#     @info(crayon"blue"("Iteration $ite done."))
# end



