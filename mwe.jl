using Test
using Enzyme
using KernelAbstractions
using CUDA
using EnzymeCore
using EnzymeCore.EnzymeRules

const KA = KernelAbstractions
function EnzymeRules.augmented_primal(
    config::Config,
    func::Const{typeof(KA.synchronize)},
    ::Type{Const{Nothing}},
    backend::T
) where T <: EnzymeCore.Annotation
    println("Forward synchronize for $(typeof(backend))")
    return AugmentedReturn{Nothing, Nothing, Any}(
        nothing, nothing, (nothing)
    )
end

function EnzymeRules.reverse(config::Config, func::Const{typeof(KA.synchronize)}, ::Type{Const{Nothing}}, tape, backend)
    println("Reverse synchronize for $(typeof(backend))")
    return (nothing,)
end
const KA = KernelAbstractions

# @kernel function square!(A,B)
#     I = @index(Global, Linear)
#     @inbounds A[I] = B[I] * B[I]
#     @synchronize()
# end

function square_caller(A, B, backend)
    # kernel = square!(backend)
    # kernel(A, B, ndrange=size(A))
    KA.synchronize(backend)
    return nothing
end

function enzyme_testsuite(backend, ArrayT)
    A = ArrayT(zeros(64))
    dA = ArrayT{Float64}(undef, 64)
    B = ArrayT{Float64}(undef, 64)
    dB = ArrayT{Float64}(undef, 64)

    dA .= 1
    B .= (1:1:64)
    dB .= 1
    x = [2.0]
    dx = [0.0]
    # Enzyme.autodiff(
    #     ReverseWithPrimal, square_caller, Duplicated(A, dA),
    #     Duplicated(B, dB), Duplicated(x,dx)
    # )
    Enzyme.autodiff(
        ReverseWithPrimal, square_caller, Duplicated(A, dA),
        Duplicated(B, dB), Const(backend())
    )
    KA.synchronize(backend())
end

# Doesn't trigger rules above
enzyme_testsuite(CPU, Array)
# Triggers rules above
enzyme_testsuite(CUDABackend, CuArray)
