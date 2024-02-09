using Test
using Enzyme
using KernelAbstractions
using CUDA

@kernel function id!(A,B)
    I = @index(Global, Linear)
    @inbounds A[I] = B[I]
    @synchronize()
    return nothing
end

function id_caller(A, B, backend)
    kernel = id_square!(backend)
    kernel(A, B, ndrange=size(A))
    return nothing
end

function enzyme_testsuite(backend, ArrayT)
    @testset "kernels" begin
        A = CuArray(zeros(64))
        dA = ArrayT{Float64}(undef, 64)
        B = ArrayT{Float64}(undef, 64)
        dB = ArrayT{Float64}(undef, 64)

        dA .= 1
        B .= (1:1:64)
        dB .= 1

        Enzyme.autodiff(ReverseWithPrimal, id_caller, Duplicated(A, dA), Duplicated(B, dB), Const(backend()))
        KernelAbstractions.synchronize(backend())
        @show A, B
        @show dA, dB
    end
end

@assert CUDA.functional()
@assert CUDA.has_cuda_gpu()
# enzyme_testsuite(CPU, Array)
enzyme_testsuite(CUDABackend, CuArray)