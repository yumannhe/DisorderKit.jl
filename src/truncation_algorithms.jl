abstract type AbstractTruncationAlgorithm end

# Standard truncation algorithm for ordinary MPOs
struct StandardTruncation <: AbstractTruncationAlgorithm
    trunc_method::TruncationScheme
    verbosity::Int
end

StandardTruncation(; trunc_method::TruncationScheme = truncerr(1e-6), verbosity::Int = 0) = StandardTruncation(trunc_method, verbosity)

# Truncation algorithm for the disorder MPO by tracing out disorder sectors
struct DisorderTracedTruncation <: AbstractTruncationAlgorithm
    trunc_method::TruncationScheme # Method for truncating ordinary mpo
    verbosity::Int
end

DisorderTracedTruncation(; trunc_method::TruncationScheme = truncerr(1e-6), verbosity::Int = 0) = DisorderTracedTruncation(trunc_method, verbosity)

# Truncation algorithm for the disorder MPO with open disorder sectors
struct DisorderOpenTruncation <: AbstractTruncationAlgorithm
    trunc_method::TruncationScheme # Method for truncating ordinary mpo
    verbosity::Int
end

DisorderOpenTruncation(; trunc_method::TruncationScheme = truncerr(1e-6), verbosity::Int = 0) = DisorderOpenTruncation(trunc_method, verbosity)


# Truncation algorithm for the disorder MPO by using SVD optimization for isometries
struct SVDUpdateTruncation <: AbstractTruncationAlgorithm
    D_max::Int
    conv_tol::Float64
    f_tol::Float64
    maxit::Int
    verbosity::Int
end

SVDUpdateTruncation(D_max::Int; conv_tol::Float64 = 1e-8, f_tol::Float64, maxit::Int = 10, verbosity::Int = 0) = SVDUpdateTruncation(D_max, conv_tol, f_tol, maxit, verbosity)

struct QRTruncation <: AbstractTruncationAlgorithm
    # trunc_method::TruncationScheme # Method for truncating ordinary mpo
    # verbosity::Int
    tol::Float64
    max_iter::Int
    trunc::TruncationScheme
    truncdim::TruncationScheme
end

QRTruncation(; tol::Float64=1e-12, max_iter::Int=200, trunc::TruncationScheme=truncbelow(1e-10), truncdim::TruncationScheme=truncerr(1e-6)) = QRTruncation(tol, max_iter, trunc, truncdim)
