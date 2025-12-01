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
    L_new = permute(S*Vt, ((1,),(2,)))
    ulphase, L_new = leftorth(L_new)
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
    R_new = permute(U*S, ((1,),(2,)))
    R_new, vtrphase = rightorth(R_new)
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
    U, L, diff, S = leftorth_trunc(L,A;trunc)
    # check if the space match
    dimL, dim_U = space(L,1), space(U,1)
    while dimL != dim_U
        U, L, diff, S = leftorth_trunc(L,A;trunc)
        dimL, dim_U = space(L,1), space(U,1)
        it +=1
    end
    while (diff > conv_tol) && (it < max_iter)
        # @show space(L), space(A), space(U)
        transferl = fixed_point_left(A, U)
        # transferl = fixed_point_left(A, LA)
        _, L = eigsolve(transferl, L, 1, :LM; tol = diff/10)
        _, L = leftorth(L[1])
        L = L/norm(L)
        U, L, diff, S = leftorth_trunc(L,A;trunc)
        @show space(L),diff
        # check if the space match
        dimL, dim_U = space(L,1), space(U,1)
        it +=1
        while dimL != dim_U
            U, L, diff, S = leftorth_trunc(L,A;trunc)
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
    Vt, R, diff, S = rightorth_trunc(R,A;trunc)
    #check the dimension
    dimR, dim_Vt = space(R,2), space(Vt,6)
    while dimR != dim_Vt
        Vt, R, diff, S = rightorth_trunc(R,A;trunc)
        #check the dimension
        dimR, dim_Vt = space(R,2), space(Vt,6)
        it +=1
    end
    while (diff > conv_tol) && (it < max_iter)
        # @show space(R), space(A), space(Vt)
        transferr = fixed_point_right(A, Vt)
        _, R = eigsolve(transferr, R, 1, :LM; tol = diff/10)
        # I think QR should be enough here. 
        R,_ = rightorth(R[1], ((1,),(2,)))
        R = R/norm(R)
        Vt, R, diff, S = rightorth_trunc(R, A;trunc)
        @show space(R),diff
        dimR, dim_Vt = space(R,2), space(Vt,6)
        it +=1
        while dimR != dim_Vt    
            Vt, R, diff, S = rightorth_trunc(R,A;trunc)
            #check the dimension
            dimR, dim_Vt = space(R,2), space(Vt,6)
            it +=1
        end
    end
    @show it, space(S,1)
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return Vt,R,S
end


function iterate_lc_svd(L, A, U; conv_tol=1e-12, max_iter=20, trunc=truncbelow(1e-10))
    it = 0
    dimL, dim_U = space(L,1), space(U,1)
    @tensor L_m[-1;-2] := conj(L[1;-1]) * L[1;-2]
    diff = 1
    S = nothing
    while dimL != dim_U
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
        U, S, Vt = tsvd(A_mul, ((1,2,3,4,5),(6,));trunc)
        L_new = permute(S*Vt, ((1,),(2,)))
        L_new = L_new/norm(L_new)
        @tensor L_new_m[-1;-2] := conj(L_new[1;-1]) * L_new[1;-2]
        dimL, dim_U = space(L_new,1), space(U,1)
        diff = norm(L_m - L_new_m)
        it +=1
        L = L_new
        L_m = L_new_m
    end
    while (diff > conv_tol) && (it < max_iter)
        # @show space(L), space(A), space(U)
        transferl = fixed_point_left(A, U)
        # transferl = fixed_point_left(A, LA)
        _, L = eigsolve(transferl, L, 1, :LM; tol = diff/10)
        _, L = leftorth(L[1])
        L = L/norm(L)
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
        U, S, Vt = tsvd(A_mul, ((1,2,3,4,5),(6,));trunc)
        L_new = permute(S*Vt, ((1,),(2,)))
        L_new = L_new/norm(L_new)
        @tensor L_new_m[-1;-2] := conj(L_new[1;-1]) * L_new[1;-2]
        diff = norm(L_new_m - L_m)
        @show space(L),diff
        # check if the space match
        # dimL, dim_U = space(L,1), space(U,1)
        it +=1
        L = L_new
        L_m = L_new_m
        # while dimL != dim_U
        #     U, L, diff, S = leftorth_trunc(L,A;trunc)
        #     dimL, dim_U = space(L,1), space(U,1)
        #     it +=1
        # end
    end
    @show it, space(S,1)
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return U,L,S
end


function iterate_rc_svd(R, A, Vt; conv_tol=1e-12, max_iter=20, trunc=truncbelow(1e-10))
    it = 0
    S = nothing
    dimR, dim_Vt = space(R,2), space(Vt,6)
    @tensor R_m[-1;-2] := conj(R[-2;1]) * R[-1;1]
    diff = 1
    while dimR != dim_Vt
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := A[-1 -2 -3;-4 -5 1] * R[1;-6]
        U, S, Vt = tsvd(A_mul, ((1,),(2,3,4,5,6));trunc)
        R_new = permute(U*S, ((1,),(2,)))
        R_new = R_new / norm(R_new)
        @tensor R_new_m[-1;-2] := R_new[-1;1] * conj(R_new[-2;1])
        dimR, dim_Vt = space(R_new,2), space(Vt,6)
        diff = norm(R_m - R_new_m)
        it +=1
        R = R_new
        R_m = R_new_m
    end
    while (diff > conv_tol) && (it < max_iter)
        # @show space(L), space(A), space(U)
        transferr = fixed_point_right(A, Vt)
        # transferl = fixed_point_left(A, LA)
        _, R = eigsolve(transferr, R, 1, :LM; tol = diff/10)
        R, _ = rightorth(R[1])
        R = R/norm(R)
        @tensor A_mul[-1 -2 -3;-4 -5 -6] := A[-1 -2 -3;-4 -5 1] * R[1;-6]
        U, S, Vt = tsvd(A_mul, ((1,),(2,3,4,5,6));trunc)
        R_new = permute(U*S, ((1,),(2,)))
        R_new = R_new / norm(R_new)
        @tensor R_new_m[-1;-2] := R_new[-1;1] * conj(R_new[-2;1])
        diff = norm(R_m - R_new_m)
        @show space(R),diff
        # check if the space match
        # dimL, dim_U = space(L,1), space(U,1)
        it +=1
        R = R_new
        R_m = R_new_m
        # while dimL != dim_U
        #     U, L, diff, S = leftorth_trunc(L,A;trunc)
        #     dimL, dim_U = space(L,1), space(U,1)
        #     it +=1
        # end
    end
    @show it, space(S,1)
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return Vt,R,S
end


partitionL = ((1,2,3,4,5,),(6,))
partitionR = ((1,),(2,3,4,5,6,))
filename = "rho_step1.jld2"
ρlist = load(filename, "ρsub")
trunc1 = truncbelow(1e-10)
ite_keeper = length(ρlist)
# truncfinal = truncdim(D_max)
for ite in 1:ite_keeper
    ρ = ρlist[ite][1]
    # initialize L and R, first check the unnormalized case
    @show space(ρ)
    Uρl, Sρl, Vtρl = tsvd(ρ, partitionL;trunc=trunc1)
    @show space(Sρl)
    Lρp = Sρl*Vtρl
    ulphase, Lρ = leftorth(Lρp)
    Uρl = Uρl*ulphase
    Lρ = Lρ / norm(Lρ)
    Uρr, Sρr, Vtρr = tsvd(ρ, partitionR;)
    @show space(Sρr)
    Rρp = Uρr*Sρr
    Rρ, urphase = rightorth(Rρp)
    Vtρr = urphase*Vtρr
    Rρ = Rρ / norm(Rρ)
    t_lc_svd = @elapsed begin
        Uρ, Lρfinal, S_Lρ = iterate_lc(Lρ, ρ;trunc=trunc1)
    end
    println("SVDQR Left took $(t_lc_svd) seconds") 
    t_rc_svd = @elapsed begin
        Vtρ, Rρfinal, S_Rρ = iterate_rc(Rρ, ρ;)
    end
    println("SVDQR Right took $(t_rc_svd) seconds") 
    # check the LR and entanglement spectrum 
    Cρ = Lρfinal*Rρfinal
    # UL, Sρ, VtR = tsvd(Cρ; trunc=truncfinal)
    UL, Sρ, VtR = tsvd(Cρ, ((1,),(2,));trunc=trunc1)
    d = dim(space(Sρ,1))
    @show d, Sρ[d,d]
    @tensor ALρ[-1 -2 -3; -4 -5 -6] := conj(UL[1 ;-1]) * Uρ[1 -2 -3;-4 -5 2] * UL[2;-6]
    @tensor ARρ[-1 -2 -3; -4 -5 -6] := VtR[-1 ;1] * Vtρ[1 -2 -3;-4 -5 2] * conj(VtR[-6;2])
    # @tensor ALρ[-1 -2 -3; -4 -5 -6] := Uρ[-1 -2 -3;-4 -5 2] * UL[2;-6]
    # @tensor ARρ[-1 -2 -3; -4 -5 -6] := VtR[-1 ;1] * Vtρ[1 -2 -3;-4 -5 -6]
    # # fidelity check:
    @tensor ALC[-1 -2 -3; -4 -5 -6] := ALρ[-1 -2 -3; -4 -5 1] * Sρ[1;-6]
    @tensor ARC[-1 -2 -3; -4 -5 -6] := Sρ[-1;1] * ARρ[1 -2 -3; -4 -5 -6]
    fidelity= norm(ALC-ARC)
    @info(crayon"blue"("Iteration $ite: Fidelity between left and right canonical forms = $fidelity"))
end


for ite in 1:ite_keeper
    ρ = ρlist[ite][1]
    # initialize L and R, first check the unnormalized case
    @show space(ρ)
    Uρl, Sρl, Vtρl = tsvd(ρ, partitionL;trunc=trunc1)
    @show space(Sρl)
    Lρ = Sρl*Vtρl
    Lρ = Lρ / norm(Lρ)
    Uρr, Sρr, Vtρr = tsvd(ρ, partitionR;trunc=trunc1)
    @show space(Sρr)
    Rρ = Uρr*Sρr
    Rρ = Rρ / norm(Rρ)
    t_lc_svd = @elapsed begin
        Uρ, Lρfinal, S_Lρ = iterate_lc_svd(Lρ, ρ, Uρl;)
    end
    println("SVD Left took $(t_lc_svd) seconds") 
    t_rc_svd = @elapsed begin
        Vtρ, Rρfinal, S_Rρ = iterate_rc_svd(Rρ, ρ, Vtρr;)
    end
    println("SVD Right took $(t_rc_svd) seconds") 
    # check the LR and entanglement spectrum 
    Cρ = Lρfinal*Rρfinal
    # UL, Sρ, VtR = tsvd(Cρ; trunc=truncfinal)
    UL, Sρ, VtR = tsvd(Cρ, ((1,),(2,)); trunc=trunc1)
    d = dim(space(Sρ,1))
    @show d, Sρ[d,d]
    # @tensor ALρ[-1 -2 -3; -4 -5 -6] := conj(UL[1 ;-1]) * Uρ[1 -2 -3;-4 -5 2] * UL[2;-6]
    # @tensor ARρ[-1 -2 -3; -4 -5 -6] := VtR[-1 ;1] * Vtρ[1 -2 -3;-4 -5 2] * conj(VtR[-6;2])
    # # @tensor ALρ[-1 -2 -3; -4 -5 -6] := Uρ[-1 -2 -3;-4 -5 2] * UL[2;-6]
    # # @tensor ARρ[-1 -2 -3; -4 -5 -6] := VtR[-1 ;1] * Vtρ[1 -2 -3;-4 -5 -6]
    # # # fidelity check:
    # @tensor ALC[-1 -2 -3; -4 -5 -6] := ALρ[-1 -2 -3; -4 -5 1] * Sρ[1;-6]
    # @tensor ARC[-1 -2 -3; -4 -5 -6] := Sρ[-1;1] * ARρ[1 -2 -3; -4 -5 -6]
    # fidelity= norm(ALC-ARC)
    @info(crayon"blue"("Iteration $ite done"))
end