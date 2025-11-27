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
function iterate_lc(L, A; conv_tol=1e-8, max_iter=100)
    @tensor A_mul[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
    U, L_new = leftorth(A_mul, ((1,2,3,4,5),(6,)))
    L_new = L_new / norm(L_new)
    diff = norm(L - L_new)
    it = 0
    L = L_new
    L_list = [L, ]
    while (diff > conv_tol) && (it < max_iter)
        transferl = fixed_point_left(A, U)
        _, L_fixed = eigsolve(transferl, L, 1, :LM; tol = diff/10)
        # @show L_fixed
        # I think QR should be enough here. 
        @show space(L_fixed[1])
        _, L = leftorth(L_fixed[1], ((1,),(2,)))
        L = L/norm(L)
        # L = L_fixed[1]/norm(L_fixed[1])
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
        U, L_new = leftorth(A_mul, ((1,2,3,4,5),(6,)))
        L_new = L_new / norm(L_new)
        diff = norm(L-L_new)
        # @show it, diff
        it += 1
        L = L_new
        push!(L_list, L)
    end
    @show it
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return U,L_list
end
    

function iterate_rc(R, A; conv_tol=1e-8, max_iter=100)
    @tensor A_mul[-1 -2 -3;-4 -5 -6] := A[-1 -2 -3;-4 -5 1] * R[1;-6]
    R_new, Vt = rightorth(A_mul, ((1,),(2,3,4,5,6)))
    R_new = R_new / norm(R_new)
    diff = norm(R-R_new)
    it = 0
    R = R_new
    R_list = [R, ]
    while (diff > conv_tol) && (it < max_iter)
        transferr = fixed_point_right(A, Vt)
        _, R_fixed = eigsolve(transferr, R, 1, :LM; tol = diff/10)
        # I think QR should be enough here. 
        R,_ = rightorth(R_fixed[1], ((1,),(2,)))
        R = R/norm(R)
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := A[-1 -2 -3;-4 -5 1] * R[1;-6]
        R_new, Vt = rightorth(A_mul, ((1,),(2,3,4,5,6)))
        R_new = R_new / norm(R_new)
        diff = norm(R-R_new)
        it += 1
        R = R_new
        push!(R_list, R)
    end
    @show it
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return Vt,R_list
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
trunc = truncbelow(1e-12)
ALC_list = []
ARC_list = []
S_check_ρ = []
L_list = []
R_list = []
ite_keeper = length(ρlist)

R_list = []
# ite_keeper = length(ρlist)
ite_keeper = 1
for ite in 1:ite_keeper
    ρ = ρlist[ite][1]
    # initialize L and R, first check the unnormalized case
    Ql, Lqrρ = leftorth(ρ, partitionL)
    Lqrρ = Lqrρ/norm(Lqrρ)
    Rqrρ, Qr = rightorth(ρ, partitionR)
    Rqrρ = Rqrρ/norm(Rqrρ)
    t_lc_qr = @elapsed begin
        Uρ, Lρ_list = iterate_lc(Lqrρ, ρ;)
    end
    println("QR Left took $(t_lc_qr) seconds") 
    t_rc_qr = @elapsed begin
        Vtρ, Rρ_list = iterate_rc(Rqrρ, ρ;)
    end
    println("QR Right took $(t_rc_qr) seconds") 
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






