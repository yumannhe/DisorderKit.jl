# measure average correlation length
function sample_correlation_length(ρs::DisorderMPO, ps::Vector{<:Real}; nsamples::Int = 100)
    D_disorder = length(ps)
    unit_cell = length(ρs)

    Zs = partition_functions(ρs)

    ξs = zeros(ComplexF64, nsamples)
    Ps = zeros(ComplexF64, nsamples)
    for in in 1:nsamples
        # Sample a random disorder realization
        A = id(ComplexF64,space(Zs[1])[1])
        p = 1
        for i in 1:unit_cell
            σ = rand(1:D_disorder)
            Q = DiagonalTensorMap(zeros(ComplexF64, D_disorder),ℂ^D_disorder)
            Q[σ,σ] = 1.0
            p = p*ps[σ]
            @tensor Z_traced[-1; -2] := Zs[i][-1 1;3 -2]*Q[3;1]
            A = A*Z_traced
        end

        vr = Tensor(rand, ComplexF64, space(A, 2)')

        λs, vrs = eigsolve(x->A*x, vr, 3, :LM)
        λ1 = λs[1]
        λ2 = λs[2]
    
        
        ξ = real(unit_cell/log(λ1/λ2))
        ξs[in] = ξ
        Ps[in] = p
    end

    return ξs, Ps
end