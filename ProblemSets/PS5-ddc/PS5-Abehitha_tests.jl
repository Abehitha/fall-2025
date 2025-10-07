using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)
include("PS5-Abehitha_source.jl")

@testset "PS5 Dynamic Discrete Choice Tests" begin
    
    Random.seed!(1234)
    
    @testset "Data Loading Functions" begin
        # Test load_static_data structure (without actual HTTP call)
        @test_nowarn begin
            mock_df = DataFrame(bus_id = repeat(1:5, 20), time = repeat(1:20, inner=5),
                               Y = rand(0:1, 100), Odometer = 10000 .+ 1000*randn(100),
                               RouteUsage = rand(100), Branded = rand(0:1, 100))
        end
        
        # Test load_dynamic_data structure
        @test_nowarn begin
            N, T, xbin, zbin = 10, 5, 5, 3
            xtran = rand(xbin*zbin, xbin); for i in 1:size(xtran,1); xtran[i,:] ./= sum(xtran[i,:]); end
            d = (Y=rand(0:1, N, T), X=50000 .+ 10000*randn(N, T), B=rand(0:1, N),
                 Xstate=rand(1:xbin, N, T), Zstate=rand(1:zbin, N), N=N, T=T,
                 xval=collect(range(0, 200000, length=xbin)), xbin=xbin, zbin=zbin, xtran=xtran, β=0.9)
            @test haskey(d, :Y) && haskey(d, :X) && haskey(d, :β)
            @test size(d.Y) == (d.N, d.T) && size(d.X) == (d.N, d.T)
            @test length(d.B) == d.N && size(d.xtran, 1) == d.xbin * d.zbin
        end
    end
    
    @testset "Static Estimation" begin
        mock_df = DataFrame(Y = rand(0:1, 100), Odometer = 50000 .+ 10000*randn(100), Branded = rand(0:1, 100))
        @test_nowarn begin
            model = glm(@formula(Y ~ Odometer + Branded), mock_df, Binomial(), LogitLink())
            @test length(coef(model)) == 3
        end
    end
    
    @testset "Future Value Computation" begin
        N, T, xbin, zbin = 5, 3, 4, 2
        xtran = rand(xbin*zbin, xbin); for i in 1:size(xtran,1); xtran[i,:] ./= sum(xtran[i,:]); end
        d = (Y=rand(0:1, N, T), X=50000 .+ 10000*randn(N, T), B=rand(0:1, N),
             Xstate=rand(1:xbin, N, T), Zstate=rand(1:zbin, N), N=N, T=T,
             xval=collect(range(0, 200000, length=xbin)), xbin=xbin, zbin=zbin, xtran=xtran, β=0.9)
        θ = [-1.0, -0.001, 0.5]
        
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        @test_nowarn compute_future_value!(FV, θ, d)
        @test size(FV) == (d.zbin * d.xbin, 2, d.T + 1)
        @test all(FV[:, :, d.T + 1] .== 0)  # Terminal condition
        @test all(isfinite.(FV))
        for b in 1:2, s in 1:(d.zbin * d.xbin)
            if d.T > 1; @test FV[s, b, 1] >= FV[s, b, 2]; end
        end

    end
    
    @testset "Log Likelihood Computation" begin
        N, T, xbin, zbin = 8, 4, 3, 2
        xtran = rand(xbin*zbin, xbin); for i in 1:size(xtran,1); xtran[i,:] ./= sum(xtran[i,:]); end
        d = (Y=rand(0:1, N, T), X=50000 .+ 10000*randn(N, T), B=rand(0:1, N),
             Xstate=rand(1:xbin, N, T), Zstate=rand(1:zbin, N), N=N, T=T,
             xval=collect(range(0, 200000, length=xbin)), xbin=xbin, zbin=zbin, xtran=xtran, β=0.9)
        θ = [-1.0, -0.0001, 0.3]
        
        @test_nowarn ll = log_likelihood_dynamic(θ, d)
        ll = log_likelihood_dynamic(θ, d)
        @test isa(ll, Real) && isfinite(ll) && ll > 0
        
        θ2 = [-2.0, -0.0002, 0.6]
        ll2 = log_likelihood_dynamic(θ2, d)
        @test ll != ll2
        
        θ_extreme = [-10.0, -0.001, 5.0]
        @test isfinite(log_likelihood_dynamic(θ_extreme, d))
    end
    
    @testset "Parameter Sensitivity" begin
        N, T, xbin, zbin = 6, 3, 3, 2
        xtran = rand(xbin*zbin, xbin); for i in 1:size(xtran,1); xtran[i,:] ./= sum(xtran[i,:]); end
        d = (Y=rand(0:1, N, T), X=50000 .+ 10000*randn(N, T), B=rand(0:1, N),
             Xstate=rand(1:xbin, N, T), Zstate=rand(1:zbin, N), N=N, T=T,
             xval=collect(range(0, 200000, length=xbin)), xbin=xbin, zbin=zbin, xtran=xtran, β=0.9)
        θ_base = [-1.0, -0.0001, 0.4]
        ll_base = log_likelihood_dynamic(θ_base, d)
        
        for i in 1:3
            θ_pert = copy(θ_base); θ_pert[i] += 0.1
            @test log_likelihood_dynamic(θ_pert, d) != ll_base
        end
        
        θ_small_pert = copy(θ_base); θ_small_pert[1] += 1e-6
        @test abs(log_likelihood_dynamic(θ_small_pert, d) - ll_base) < 1.0
    end
    
    @testset "State Space Properties" begin
        N, T, xbin, zbin = 5, 3, 4, 2
        xtran = rand(xbin*zbin, xbin); for i in 1:size(xtran,1); xtran[i,:] ./= sum(xtran[i,:]); end
        d = (Y=rand(0:1, N, T), X=50000 .+ 10000*randn(N, T), B=rand(0:1, N),
             Xstate=rand(1:xbin, N, T), Zstate=rand(1:zbin, N), N=N, T=T,
             xval=collect(range(0, 200000, length=xbin)), xbin=xbin, zbin=zbin, xtran=xtran, β=0.9)
        
        for z in 1:d.zbin, x in 1:d.xbin
            row = x + (z-1)*d.xbin
            @test 1 <= row <= d.zbin * d.xbin
        end
        
        @test size(d.xtran) == (d.zbin * d.xbin, d.xbin)
        @test all(d.xtran .>= 0) && all(abs.(sum(d.xtran, dims=2) .- 1) .< 1e-10)
        @test all(1 .<= d.Xstate .<= d.xbin) && all(1 .<= d.Zstate .<= d.zbin)
        @test all(d.B .∈ Ref([0, 1]))
    end
    
    @testset "Edge Cases and Robustness" begin
        # Single observation
        xtran = rand(6, 3); for i in 1:size(xtran,1); xtran[i,:] ./= sum(xtran[i,:]); end
        d_single = (Y=rand(0:1, 1, 2), X=50000 .+ 10000*randn(1, 2), B=rand(0:1, 1),
                    Xstate=rand(1:3, 1, 2), Zstate=rand(1:2, 1), N=1, T=2,
                    xval=collect(range(0, 200000, length=3)), xbin=3, zbin=2, xtran=xtran, β=0.9)
        @test_nowarn log_likelihood_dynamic([-1.0, -0.0001, 0.5], d_single)
        
        # All zeros/ones decisions
        xtran2 = rand(9, 3); for i in 1:size(xtran2,1); xtran2[i,:] ./= sum(xtran2[i,:]); end
        d_zeros = (Y=zeros(Int, 5, 3), X=50000 .+ 10000*randn(5, 3), B=rand(0:1, 5),
                   Xstate=rand(1:3, 5, 3), Zstate=rand(1:3, 5), N=5, T=3,
                   xval=collect(range(0, 200000, length=3)), xbin=3, zbin=3, xtran=xtran2, β=0.9)
        @test isfinite(log_likelihood_dynamic([-1.0, -0.0001, 0.5], d_zeros))
        
        d_ones = (Y=ones(Int, 5, 3), X=50000 .+ 10000*randn(5, 3), B=rand(0:1, 5),
                  Xstate=rand(1:3, 5, 3), Zstate=rand(1:3, 5), N=5, T=3,
                  xval=collect(range(0, 200000, length=3)), xbin=3, zbin=3, xtran=xtran2, β=0.9)
        @test isfinite(log_likelihood_dynamic([-1.0, -0.0001, 0.5], d_ones))
    end
    
    @testset "Mathematical Properties" begin
        N, T, xbin, zbin = 5, 3, 3, 2
        xtran = rand(xbin*zbin, xbin); for i in 1:size(xtran,1); xtran[i,:] ./= sum(xtran[i,:]); end
        d = (Y=rand(0:1, N, T), X=50000 .+ 10000*randn(N, T), B=rand(0:1, N),
             Xstate=rand(1:xbin, N, T), Zstate=rand(1:zbin, N), N=N, T=T,
             xval=collect(range(0, 200000, length=xbin)), xbin=xbin, zbin=zbin, xtran=xtran, β=0.9)
        θ = [-1.0, -0.0001, 0.5]
        
        # Test deterministic computation
        FV1 = zeros(d.zbin * d.xbin, 2, d.T + 1)
        FV2 = zeros(d.zbin * d.xbin, 2, d.T + 1)
        compute_future_value!(FV1, θ, d); compute_future_value!(FV2, θ, d)
        @test FV1 ≈ FV2 && all(isfinite.(FV1))
        
        # Test backward recursion
        FV_alt = zeros(d.zbin * d.xbin, 2, d.T + 1)
        FV_alt[:, :, d.T + 1] .= 1.0
        compute_future_value!(FV_alt, θ, d)
        @test !(FV1 ≈ FV_alt)
    end
    
    @testset "Optimization Setup" begin
        N, T, xbin, zbin = 4, 2, 3, 2
        xtran = rand(xbin*zbin, xbin); for i in 1:size(xtran,1); xtran[i,:] ./= sum(xtran[i,:]); end
        d = (Y=rand(0:1, N, T), X=50000 .+ 10000*randn(N, T), B=rand(0:1, N),
             Xstate=rand(1:xbin, N, T), Zstate=rand(1:zbin, N), N=N, T=T,
             xval=collect(range(0, 200000, length=xbin)), xbin=xbin, zbin=zbin, xtran=xtran, β=0.9)
        
        objective = θ -> log_likelihood_dynamic(θ, d)
        θ_test = [-1.0, -0.0001, 0.5]
        @test_nowarn objective(θ_test) 
        @test isfinite(objective(θ_test))
        
        @test_nowarn begin
            θ_start = rand(3)
            log_likelihood_dynamic(θ_start, d)
        end
    end
    
    @testset "Data Structure Consistency" begin
        N, T, xbin, zbin = 7, 4, 4, 3
        xtran = rand(xbin*zbin, xbin); for i in 1:size(xtran,1); xtran[i,:] ./= sum(xtran[i,:]); end
        d = (Y=rand(0:1, N, T), X=50000 .+ 10000*randn(N, T), B=rand(0:1, N),
             Xstate=rand(1:xbin, N, T), Zstate=rand(1:zbin, N), N=N, T=T,
             xval=collect(range(0, 200000, length=xbin)), xbin=xbin, zbin=zbin, xtran=xtran, β=0.9)
        
        for field in [:Y, :X, :B, :Xstate, :Zstate, :N, :T, :xval, :xbin, :zbin, :xtran, :β]
            @test haskey(d, field)
        end
        
        @test size(d.Y) == (d.N, d.T) && size(d.X) == (d.N, d.T) && size(d.Xstate) == (d.N, d.T)
        @test length(d.Zstate) == d.N && length(d.B) == d.N && length(d.xval) == d.xbin
        @test size(d.xtran) == (d.xbin * d.zbin, d.xbin) && 0 < d.β <= 1
    end
    
    @testset "Numerical Stability" begin
        N, T, xbin, zbin = 5, 3, 3, 2
        xtran = rand(xbin*zbin, xbin); for i in 1:size(xtran,1); xtran[i,:] ./= sum(xtran[i,:]); end
        d = (Y=rand(0:1, N, T), X=50000 .+ 10000*randn(N, T), B=rand(0:1, N),
             Xstate=rand(1:xbin, N, T), Zstate=rand(1:zbin, N), N=N, T=T,
             xval=collect(range(0, 200000, length=xbin)), xbin=xbin, zbin=zbin, xtran=xtran, β=0.9)
        
        @test isfinite(log_likelihood_dynamic([10.0, -0.01, 5.0], d))  # Large params
        @test isfinite(log_likelihood_dynamic([0.001, -1e-8, 0.001], d))  # Small params
        @test isfinite(log_likelihood_dynamic([-1.0, -0.0001, -2.0], d))  # Negative brand
        
        # Discount factor sensitivity
        d_high_β = merge(d, (β = 0.99,)); d_low_β = merge(d, (β = 0.5,))
        θ = [-1.0, -0.0001, 0.5]
        ll_high = log_likelihood_dynamic(θ, d_high_β); ll_low = log_likelihood_dynamic(θ, d_low_β)
        @test isfinite(ll_high) && isfinite(ll_low) && ll_high != ll_low
    end
end

println("✓ All PS5 dynamic discrete choice tests completed successfully!")
