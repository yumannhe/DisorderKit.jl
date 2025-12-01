# Compute truncation matrices
function truncation_matrices(M::InfiniteMPO, trunc_method::TruncationScheme)
    L = length(M)

    envLs = map(ix -> env_left(M, ix), 1:L)
    envRs = map(ix -> env_right(M, ix), 1:L)

    Xs = map(envLs) do ρL
        _, SL, VL = tsvd(ρL; trunc=truncerr(1e-12));
        X = sqrt(SL) * VL
        Xinv = VL' * inv(sqrt(SL))
        return (X, Xinv)
    end

    Ys = map(envRs) do ρR
        UR, SR, _ = tsvd(ρR; trunc=truncerr(1e-12));
        Y = UR * sqrt(SR)
        Yinv = inv(sqrt(SR)) * UR'
        return (Y, Yinv)
    end

    truncations = map(1:L) do ix
        X, Xinv = Xs[ix]
        Y, Yinv = Ys[ix]
        U, S, V = tsvd(X*Y; trunc=trunc_method)
        PL = sqrt(S) * V * Yinv
        PR = Xinv * U * sqrt(S)
        return (PL, PR)
    end

    return PeriodicVector(truncations)
end

# Truncate ordinary mpo with standard truncation algorithm
function truncate_mpo(mpo::InfiniteMPO, alg::StandardTruncation)
    truncations = truncation_matrices(mpo, alg.trunc_method)
    L = length(mpo)
    mpo_updated = map(1:L) do ix
        PL = truncations[ix-1][1]
        PR = truncations[ix][2]
        @tensor O_updated[-1 -2 ; -3 -4] := PL[-1; 1] * mpo[ix][1 -2; -3 2] * PR[2; -4]
        return O_updated
    end
    return InfiniteMPO(mpo_updated)
end

# Truncate DisorderMPO by tracing disorder sectors
function truncate_mpo(ρ::DisorderMPO, ps::Vector{<:Real}, alg::DisorderTracedTruncation)
    @info(crayon"red"("Truncate DisorderMPO"))
    ρn_weighted = disorder_average(ρ, ps)
    
    truncations = truncation_matrices(ρn_weighted, alg.trunc_method)
    L = length(ρn_weighted)
    ρs_updated = map(1:L) do ix
        PL = truncations[ix-1][1]
        PR = truncations[ix][2]
        @tensor ρ1_updated[-1 -2 -3; -4 -5 -6] := PL[-1; 1] * ρ[ix][1 -2 -3; -4 -5 2] * PR[2; -6]
        return ρ1_updated
    end

    return DisorderMPO(ρs_updated)
end

# Truncate DisorderMPO with open disorder sectors
function truncate_mpo(ρ::DisorderMPO, ps::Vector{<:Real}, alg::DisorderOpenTruncation)
    @info(crayon"red"("Truncate DisorderMPO"))
    L = length(ρ)
    ρs_fused = map(1:L) do ix
        sp1, sp2 = space(ρ[ix], 2), space(ρ[ix], 3)
        iso = isomorphism(fuse(sp1, sp2), sp1*sp2)
        @tensor ρ1_updated[-1 -2; -3 -4] := iso[-2; 1 2] * ρ[ix][-1 1 2; 3 4 -4] * conj(iso[-3; 3 4])
        return ρ1_updated
    end
    ρn_weighted = InfiniteMPO(ρs_fused)
    
    truncations = truncation_matrices(ρn_weighted, alg.trunc_method)
    L = length(ρn_weighted)
    ρs_updated = map(1:L) do ix
        PL = truncations[ix-1][1]
        PR = truncations[ix][2]
        @tensor ρ1_updated[-1 -2 -3; -4 -5 -6] := PL[-1; 1] * ρ[ix][1 -2 -3; -4 -5 2] * PR[2; -6]
        return ρ1_updated
    end

    return DisorderMPO(ρs_updated)
end

# Truncate DisorderMPO by optimizing isometries with SVD update
function truncate_mpo(ρ::DisorderMPO, ps::Vector{<:Real}, alg::SVDUpdateTruncation)
    @info(crayon"red"("Truncate DisorderMPO"))
    (length(ρ)>1) && error("Only single unit cell is implemented.")
    if dim(space(ρ[1])[1])<= alg.D_max
        @warn("DisorderMPO is already truncated to the maximum bond dimension.")
        return ρ
    else
        X = TensorMap(rand,ComplexF64, ℂ^alg.D_max, space(ρ[1])[1])
        U, _, V = tsvd(X)
        X = U*V
        X_new = optimize_isometry(ρ, X, ps; conv_tol = alg.conv_tol, f_tol = alg.f_tol, maxit = alg.maxit)
        @tensor ρs_updated[-1 -2 -3; -4 -5 -6] := ρ[1][1 -2 -3; -4 -5 2] * X_new[-1; 1] * conj(X_new[-6; 2])
        return DisorderMPO([ρs_updated])
    end
end


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


function leftorth_trunc(L,A;trunc=trunc)
    @tensor A_mul[-1 -2 -3;-4 -5 -6] := L[-1;1]*A[1 -2 -3;-4 -5 -6]
    A_mul = permute(A_mul, ((1,2,3,4,5),(6,));copy = false)
    # U, S, Vt = tsvd(A_mul, ((1,2,3,4,5),(6,)); trunc=trunc)
    U, S, Vt = tsvd!(A_mul; trunc=trunc)
    L_new = permute(S*Vt, ((1,),(2,));copy = false)
    ulphase, L_new = leftorth!(L_new)
    U = permute(U*ulphase, ((1,2,3,),(4,5,6,));copy = false)
    L_new = L_new / norm(L_new)
    @tensor L_m[-1;-2] := conj(L[1;-1]) * L[1;-2]
    @tensor L_new_m[-1;-2] := conj(L_new[1;-1]) * L_new[1;-2]
    diff = norm(L_m - L_new_m)
    return U, L_new, diff,S
end


function rightorth_trunc(R,A;trunc=trunc)
    @tensor A_mul[-1 -2 -3;-4 -5 -6] := A[-1 -2 -3;-4 -5 1] * R[1;-6]
    A_mul = permute(A_mul,((1,),(2,3,4,5,6));copy = false)
    # U, S, Vt = tsvd(A_mul, ((1,),(2,3,4,5,6)); trunc=trunc)
    U, S, Vt = tsvd!(A_mul; trunc=trunc)
    R_new = permute(U*S, ((1,),(2,));copy = false)
    R_new, vtrphase = rightorth!(R_new)
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
    @show it,diff
    while (diff > conv_tol) && (it < max_iter)
        # @show space(L), space(A), space(U)
        transferl = fixed_point_left(A, U)
        # transferl = fixed_point_left(A, LA)
        _, L = eigsolve(transferl, L, 1, :LM; tol = diff/10)
        _, L = leftorth!(L[1])
        L = L/norm(L)
        U, L, diff, S = leftorth_trunc(L,A;trunc)
        
        # check if the space match
        dimL, dim_U = space(L,1), space(U,1)
        it +=1
        while dimL != dim_U
            U, L, diff, S = leftorth_trunc(L,A;trunc)
            dimL, dim_U = space(L,1), space(U,1)
            it +=1
        end
        @show it,diff
    end
    # @show it, space(S,1)
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return U,L
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
    @show it,diff
    while (diff > conv_tol) && (it < max_iter)
        # @show space(R), space(A), space(Vt)
        transferr = fixed_point_right(A, Vt)
        _, R = eigsolve(transferr, R, 1, :LM; tol = diff/10)
        # I think QR should be enough here. 
        R,_ = rightorth!(R[1])
        R = R/norm(R)
        Vt, R, diff, S = rightorth_trunc(R, A;trunc)
        dimR, dim_Vt = space(R,2), space(Vt,6)
        it +=1
        while dimR != dim_Vt    
            Vt, R, diff, S = rightorth_trunc(R,A;trunc)
            #check the dimension
            dimR, dim_Vt = space(R,2), space(Vt,6)
            it +=1
        end
        @show it,diff
    end
    # @show it, space(S,1)
    diff < conv_tol || @warn(crayon"red"("Iteration not accurate: diff = $diff"))  
    return Vt,R
end



# # truncate MPO with the QR method
# function truncate_mpo(ρ::DisorderMPO, ps::Vector{<:Real}, alg::QRTruncation)
#     conv_tol=alg.tol 
#     max_iter=alg.max_iter 
#     trunc=alg.trunc 
#     truncdim = alg.truncdim
#     Uρl, Sρl, Vtρl = tsvd(ρ[1], ((1,2,3,4,5),(6,));trunc)
#     # @show space(Sρl)
#     Lρp = Sρl*Vtρl
#     ulphase, Lρ = leftorth(Lρp)
#     Uρl = Uρl*ulphase
#     Lρ = Lρ / norm(Lρ)
#     Uρr, Sρr, Vtρr = tsvd(ρ[1], ((1,),(2,3,4,5,6));trunc)
#     # @show space(Sρr)
#     Rρp = Uρr*Sρr
#     Rρ, urphase = rightorth(Rρp)
#     Vtρr = urphase*Vtρr
#     Rρ = Rρ / norm(Rρ)
#     Uρ, Lρfinal = iterate_lc(Lρ, ρ[1];conv_tol, max_iter, trunc)
#     Vtρ, Rρfinal = iterate_rc(Rρ, ρ[1];conv_tol, max_iter, trunc)
#     Cρ = Lρfinal*Rρfinal
#     # UL, Sρ, VtR = tsvd(Cρ; trunc=truncfinal)
#     UL, Sρ, VtR = tsvd(Cρ; trunc=truncdim)
#     @tensor ALρ[-1 -2 -3; -4 -5 -6] := conj(UL[1 ;-1]) * Uρ[1 -2 -3;-4 -5 2] * UL[2;-6]
#     @tensor ARρ[-1 -2 -3; -4 -5 -6] := VtR[-1 ;1] * Vtρ[1 -2 -3;-4 -5 2] * conj(VtR[-6;2])
#     # output the left one first
#     return DisorderMPO([ALρ])
# end


function truncate_mpo(ρ::DisorderMPO, ps::Vector{<:Real}, alg::QRTruncation)
    conv_tol=alg.tol 
    max_iter=alg.max_iter 
    trunc=alg.trunc 
    truncdim = alg.truncdim
    tol_trial = √conv_tol
    # tol_trial = conv_tol
    transferL = fixed_point_left(ρ[1], ρ[1])
    transferR = fixed_point_right(ρ[1], ρ[1])
    # initialize a random tensor for finding the l/r
    xl = TensorMap(randn, ComplexF64, space(ρ[1],1),space(ρ[1],1))
    xr = TensorMap(randn, ComplexF64, space(ρ[1],6)',space(ρ[1],6)')
    _, envL = eigsolve(transferL, xl, 1, :LM; tol=tol_trial)
    _, envR = eigsolve(transferR, xr, 1, :LM; tol=tol_trial)
    Uρl, Sρl, Vtρl = tsvd!(envL[1];trunc)
    # @show space(Sρl)
    Lρp = Sρl*Vtρl
    ulphase, Lρ = leftorth!(Lρp)
    # Uρl = Uρl*ulphase
    Lρ = Lρ / norm(Lρ)
    Uρr, Sρr, Vtρr = tsvd!(envR[1];trunc)
    # @show space(Sρr)
    Rρp = Uρr*Sρr
    Rρ, urphase = rightorth!(Rρp)
    # Vtρr = urphase*Vtρr
    Rρ = Rρ / norm(Rρ)
    Uρ, Lρfinal = iterate_lc(Lρ, ρ[1];conv_tol, max_iter, trunc)
    Vtρ, Rρfinal = iterate_rc(Rρ, ρ[1];conv_tol, max_iter, trunc)
    Cρ = Lρfinal*Rρfinal
    # UL, Sρ, VtR = tsvd(Cρ; trunc=truncfinal)
    UL, Sρ, VtR = tsvd(Cρ; trunc=truncdim)
    @tensor ALρ[-1 -2 -3; -4 -5 -6] := conj(UL[1 ;-1]) * Uρ[1 -2 -3;-4 -5 2] * UL[2;-6]
    @tensor ARρ[-1 -2 -3; -4 -5 -6] := VtR[-1 ;1] * Vtρ[1 -2 -3;-4 -5 2] * conj(VtR[-6;2])
    # output the left one first
    return DisorderMPO([ALρ])
end