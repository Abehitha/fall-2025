using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions
cd(@__DIR__)
Random.seed!(1234)
include("PS4-Abehitha_source.jl")

@testset "PS4 Mixture Model Tests" begin
    
    # Test data setup
    N, K, J = 50, 3, 8
    X_test = randn(N, K)
    Z_test = randn(N, J) 
    y_test = rand(1:J, N)
    theta_test = [randn(K*(J-1)); 0.1]  # 22 params: 21 alphas + 1 gamma
    
    @testset "Data Loading" begin
        # Mock data test
        mock_df = DataFrame(
            :age =>rand(20:60, 10), 
            :white =>rand(0:1, 10), 
            :collgrad =>rand(0:1, 10),
            :occ_code =>rand(1:8, 10)
        )                           
        for i in 1:8
            mock_df[!, Symbol("elnwage$i")] = randn(10)
        end
        CSV.write("test_data.csv", mock_df)
        
        @test_nowarn begin
            df = CSV.read("test_data.csv", DataFrame)
            X = [df.age df.white df.collgrad]
            Z = hcat([df[!, Symbol("elnwage$i")] for i in 1:8]...)
            y = df.occ_code
        end
        rm("test_data.csv")
    end
    
    @testset "Multinomial Logit" begin
        @test_nowarn mlogit_with_Z(theta_test, X_test, Z_test, y_test)
        ll = mlogit_with_Z(theta_test, X_test, Z_test, y_test)
        @test ll > 0 && isfinite(ll)
        @test length(theta_test) == K*(J-1) + 1
        
        # Test probability properties
        alpha, gamma = theta_test[1:end-1], theta_test[end]
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        num = [exp.(X_test * bigAlpha[:,j] .+ gamma .* (Z_test[:,j] .- Z_test[:,J])) for j in 1:J]
        P = hcat(num...) ./ sum(hcat(num...), dims=2)
        @test all(abs.(sum(P, dims=2) .- 1) .< 1e-10)  # Probabilities sum to 1
        @test all(P .>= 0)  # Non-negative probabilities
    end
    
    @testset "Quadrature Functions" begin
        @test_nowarn lgwt(7, -4, 4)
        nodes, weights = lgwt(7, -4, 4)
        @test length(nodes) == length(weights) == 7
        @test all(isfinite.(nodes)) && all(isfinite.(weights))
        
        # Test quadrature accuracy with standard normal
        d = Normal(0, 1)
        integral = sum(weights .* pdf.(d, nodes))
        @test abs(integral - 1.0) < 0.1  # Should integrate to ≈1
        
        @test_nowarn practice_quadrature()
        @test_nowarn variance_quadrature()
        @test_nowarn practice_monte_carlo()
    end
    
    @testset "Mixed Logit Functions" begin
        theta_mixed = [randn(K*(J-1)); 0.1; 1.0]  # 23 params: 21 alphas + mu_gamma + sigma_gamma
        R, D = 5, 100  # Small values for testing
        
        @test_nowarn mixed_logit_quad(theta_mixed, X_test, Z_test, y_test, R)
        @test_nowarn mixed_logit_mc(theta_mixed, X_test, Z_test, y_test, D)
        
        ll_quad = mixed_logit_quad(theta_mixed, X_test, Z_test, y_test, R)
        ll_mc = mixed_logit_mc(theta_mixed, X_test, Z_test, y_test, D)
        @test all(isfinite.([ll_quad, ll_mc])) && all([ll_quad, ll_mc] .> 0)
        @test length(theta_mixed) == K*(J-1) + 2
    end
    
    @testset "Edge Cases & Robustness" begin
        
        # Extreme parameters
        theta_extreme = [fill(10.0, K*(J-1)); 5.0]
        @test isfinite(mlogit_with_Z(theta_extreme, X_test, Z_test, y_test))
        
        # All choices present
        y_complete = repeat(1:J, outer=ceil(Int, N/J))[1:N]
        @test_nowarn mlogit_with_Z(theta_test, X_test, Z_test, y_complete)
        
        # Zero parameters
        theta_zero = zeros(K*(J-1) + 1)
        @test isfinite(mlogit_with_Z(theta_zero, X_test, Z_test, y_test))
    end
    
    @testset "Integration & Workflow" begin
        
        # Parameter extraction test
        alpha = theta_test[1:end-1]
        gamma = theta_test[end]
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        @test size(bigAlpha) == (K, J)
        @test bigAlpha[:, end] == zeros(K)  # Last column should be zeros
        
        # Choice indicator matrix test  
        bigY = zeros(N, J)
        for j = 1:J; bigY[:, j] = y_test .== j; end
        @test all(sum(bigY, dims=2) .== 1)  # Each row sums to 1
        @test all(bigY .∈ Ref([0, 1]))  # Only 0s and 1s
    end
    
    @testset "Numerical Stability" begin
        # Large numbers
        X_large = 10 * randn(N, K)
        @test isfinite(mlogit_with_Z(theta_test, X_large, Z_test, y_test))
        
        # Different scales
        for scale in [0.01, 1.0, 10.0]
            theta_scaled = scale * theta_test
            ll_scaled = mlogit_with_Z(theta_scaled, X_test, Z_test, y_test)
            @test isfinite(ll_scaled)
        end
        
        # Mixed logit with extreme values
        theta_mixed_extreme = [fill(5.0, K*(J-1)); 2.0; 0.1]
        @test isfinite(mixed_logit_quad(theta_mixed_extreme, X_test, Z_test, y_test, 3))
        @test isfinite(mixed_logit_mc(theta_mixed_extreme, X_test, Z_test, y_test, 50))
    end
end

println("✓ All PS4 mixture model tests completed successfully!")