using Test
using Enzyme
using KernelAbstractions
using CUDA
using EnzymeCore
using EnzymeCore.EnzymeRules

const KA = KernelAbstractions

struct MyData
    A::Array{Float64}
    B::Array{Float64}
end

@kernel function square!(A,B)
    I = @index(Global, Linear)
    @inbounds A[I] = B[I] * B[I]
    @synchronize()
end

function square_caller(data, backend)
    kernel = square!(backend)
    kernel(data.A, data.B, ndrange=size(data.A))
    KA.synchronize(backend)
    return nothing
end

function square_caller(A, B, backend)
    kernel = square!(backend)
    kernel(A, B, ndrange=size(A))
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
    data = MyData(A, B)
    ddata = MyData(dA, dB)
    square_caller(data, backend())
    Enzyme.autodiff(
        Reverse, square_caller, Duplicated(A, dA), Duplicated(B, dB), Const(backend())
    )
    KA.synchronize(backend())
    # Does not work
    Enzyme.autodiff(
        Reverse, square_caller, Duplicated(data, ddata), Const(backend())
    )
    KA.synchronize(backend())
    # @show ddata.B
    # @show dB
end

enzyme_testsuite(CPU, Array)
# enzyme_testsuite(CUDABackend, CuArray)
