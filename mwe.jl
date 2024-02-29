using Test
using Enzyme
using KernelAbstractions
using CUDA

const KA = KernelAbstractions

@kernel function square!(A,B)
    I = @index(Global, Linear)
    @inbounds A[I] = B[I] * B[I]
    @synchronize()
    return nothing
end

function square_caller(A, B, backend)
    kernel = square!(backend)
    kernel(A, B, ndrange=size(A))
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
    KA.synchronize(backend())
    Enzyme.autodiff(ReverseWithPrimal, square_caller, Duplicated(A, dA), Duplicated(B, dB), Const(backend()))
    KA.synchronize(backend())
    @show A
    @show B
    @show dA
    @show dB
end

enzyme_testsuite(CPU, Array)
if CUDA.functional() && CUDA.has_cuda_gpu()
    enzyme_testsuite(CUDABackend, CuArray)
end
