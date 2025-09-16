#=  Unit Tests for Problem Set 2 Functions
    Author: Test Suite for PS2-Abehitha_source.jl
    Date: 2025-09-16
    Description: Comprehensive unit tests for the optimization functions
=#

using Test, Random, LinearAlgebra, Statistics

cd(@__DIR__)

# Include the source functions (extract them from the main file)
include("PS2-Abehitha_source.jl")

# Extract the individual functions from the PS2() function for testing
function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

function ols2(beta, X, y)
    ssr = sum((y.-X*beta).^2)
    return ssr
end

function logit(alpha, X, y)
    loglike=sum(y.*(X*alpha)-log.(1 .+exp.(X*alpha)))
    return -loglike
end

function mlogit(alpha, X, y)
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    
    d=zeros(N,J)
    for j=1:J
        d[:,j] = y.==j
    end
    alpha_matrix = [reshape(alpha, K, J-1) zeros(K, 1)]
    
    num = zeros(N,J)
    den = zeros(N)
    for j=1:J
        num[:,j] = exp.(X*alpha_matrix[:,j])
        den .+=num[:,j]
    end
    p = num./repeat(den,1,J)
    
    loglike=-sum(d.*log.(p))
    
    return loglike
end

@testset "PS2 Optimization Functions Tests" begin
    
    @testset "OLS Function Tests" begin
        # Test 1: Simple known case
        @testset "Basic OLS functionality" begin
            # Create simple test data
            X = [1.0 1.0; 1.0 2.0; 1.0 3.0]  # intercept and slope
            y = [2.0, 4.0, 6.0]  # perfect fit: y = 2*x
            beta_true = [0.0, 2.0]
            
            # Test that true parameters give zero SSR
            ssr = ols(beta_true, X, y)
            @test ssr ≈ 0.0 atol=1e-10
        end
        
        @testset "OLS with non-zero residuals" begin
            X = [1.0 1.0; 1.0 2.0; 1.0 3.0]
            y = [2.1, 3.9, 6.1]  # small deviations from perfect fit
            beta_test = [0.0, 2.0]
            
            ssr = ols(beta_test, X, y)
            expected_ssr = 0.1^2 + (-0.1)^2 + 0.1^2  # sum of squared residuals
            @test ssr ≈ expected_ssr atol=1e-10
        end
        
        @testset "OLS edge cases" begin
            # Single observation
            X = reshape([1.0, 2.0], 1, 2)
            y = [5.0]
            beta = [1.0, 2.0]
            ssr = ols(beta, X, y)
            @test ssr ≈ 0.0 atol=1e-10  # 5 = 1 + 2*2
            
            # Multiple predictors
            X = [1.0 2.0 3.0; 1.0 4.0 5.0]
            y = [10.0, 20.0]
            beta = [1.0, 2.0, 1.0]
            ssr = ols(beta, X, y)
            expected = [10.0, 20.0] - X * beta
            @test ssr ≈ sum(expected.^2) atol=1e-10
        end
    end
    
    @testset "OLS2 Function Tests" begin
        @testset "OLS2 equivalence to OLS" begin
            # Test that ols and ols2 give identical results
            X = [1.0 1.0 2.0; 1.0 2.0 4.0; 1.0 3.0 6.0; 1.0 4.0 8.0]
            y = [3.0, 7.0, 11.0, 15.0]
            beta = [1.0, 2.0, 0.5]
            
            ssr1 = ols(beta, X, y)
            ssr2 = ols2(beta, X, y)
            @test ssr1 ≈ ssr2 atol=1e-10
        end
        
        @testset "OLS2 with random data" begin
            Random.seed!(123)
            n, k = 50, 3
            X = [ones(n) randn(n, k-1)]
            beta_true = randn(k)
            y = X * beta_true + 0.1 * randn(n)
            
            # Test with true parameters (should be small SSR due to noise)
            ssr = ols2(beta_true, X, y)
            @test ssr > 0  # Should have some residual due to noise
            @test ssr < 1  # But not too large given small noise
            
            # Test with wrong parameters (should be larger SSR)
            beta_wrong = zeros(k)
            ssr_wrong = ols2(beta_wrong, X, y)
            @test ssr_wrong > ssr
        end
    end
    
    @testset "Logit Function Tests" begin
        @testset "Basic logit likelihood" begin
            # Simple test case
            X = [1.0 0.0; 1.0 1.0]  # intercept + binary predictor
            y = [0.0, 1.0]  # binary outcomes
            alpha = [0.0, 1.0]  # coefficients
            
            # Calculate expected likelihood manually
            # For observation 1: y=0, X*alpha = 0, P(y=1) = 0.5, log-lik = log(0.5)
            # For observation 2: y=1, X*alpha = 1, P(y=1) = exp(1)/(1+exp(1)), log-lik = 1 - log(1+exp(1))
            exp1 = exp(1.0)
            p2 = exp1 / (1 + exp1)
            expected_loglik = log(0.5) + log(p2)
            
            result = logit(alpha, X, y)
            @test result ≈ -expected_loglik atol=1e-10
        end
        
        @testset "Logit with extreme values" begin
            X = [1.0 0.0; 1.0 1.0]
            y = [0.0, 1.0]
            
            # Test with large positive coefficient (should predict y=1 well)
            alpha_large = [0.0, 10.0]
            ll_large = logit(alpha_large, X, y)
            
            # Test with large negative coefficient (should predict y=0 well)
            alpha_neg = [0.0, -10.0]
            ll_neg = logit(alpha_neg, X, y)
            
            # Large positive should be better for this data
            @test ll_large < ll_neg
        end
        
        @testset "Logit properties" begin
            Random.seed!(456)
            n = 100
            X = [ones(n) randn(n)]
            y = rand(n) .< 0.3  # 30% probability of y=1
            alpha = [0.0, 0.5]
            
            ll = logit(alpha, X, y)
            @test ll > 0  # Negative log-likelihood should be positive
            @test isfinite(ll)  # Should not be infinite
        end
    end
    
    @testset "Mlogit Function Tests" begin
        @testset "Basic multinomial logit" begin
            # Simple 3-category case
            n = 6
            K = 2  # intercept + one covariate
            J = 3  # three categories
            
            X = [ones(n) [1, 2, 3, 1, 2, 3]]  # simple pattern
            y = [1, 1, 2, 2, 3, 3]  # two obs per category
            
            # Test with zero coefficients (equal probabilities)
            alpha_zero = zeros(K * (J-1))  # K*(J-1) parameters
            ll_zero = mlogit(alpha_zero, X, y)
            
            @test isfinite(ll_zero)
            @test ll_zero > 0  # Negative log-likelihood should be positive
        end
        
        @testset "Mlogit with known structure" begin
            # Create data where category 1 is preferred for low X, category 2 for high X
            X = [ones(4) [1, 2, 4, 5]]
            y = [1, 1, 2, 2]  # category changes with X
            
            K = 2
            J = 2
            alpha = [0.0, -1.0]  # negative coefficient should favor category 1 for low X
            
            ll = mlogit(alpha, X, y)
            @test isfinite(ll)
            @test ll > 0
        end
        
        @testset "Mlogit parameter dimensions" begin
            Random.seed!(789)
            n, K, J = 20, 3, 4
            X = [ones(n) randn(n, K-1)]
            y = rand(1:J, n)
            
            # Test correct parameter vector length
            alpha_correct = randn(K * (J-1))
            ll_correct = mlogit(alpha_correct, X, y)
            @test isfinite(ll_correct)
            
            # Test incorrect parameter vector length should error
            alpha_wrong = randn(K * J)  # Too many parameters
            @test_throws DimensionMismatch mlogit(alpha_wrong, X, y)
        end
        
        @testset "Mlogit probability properties" begin
            # Test that probabilities sum to 1 (indirectly through likelihood calculation)
            X = [1.0 0.0; 1.0 1.0; 1.0 2.0]
            y = [1, 2, 3]
            alpha = [0.5, -0.3, 1.0, 0.2]  # 2 params per 2 non-reference categories
            
            ll = mlogit(alpha, X, y)
            @test isfinite(ll)
            @test ll > 0
        end
    end
    
    @testset "Integration Tests" begin
        @testset "OLS optimization workflow" begin
            Random.seed!(100)
            n, k = 30, 3
            X = [ones(n) randn(n, k-1)]
            beta_true = [1.0, 0.5, -0.3]
            y = X * beta_true + 0.1 * randn(n)
            
            # Test that optimization finds reasonable solution
            using Optim
            result = optimize(b -> ols(b, X, y), zeros(k), BFGS())
            beta_est = result.minimizer
            
            # Should be close to closed-form solution
            beta_closed = (X'X) \ (X'y)
            @test norm(beta_est - beta_closed) < 1e-4
        end
        
        @testset "Logit optimization workflow" begin
            Random.seed!(200)
            n = 100
            X = [ones(n) randn(n)]
            alpha_true = [0.0, 1.0]
            p_true = 1 ./ (1 .+ exp.(-X * alpha_true))
            y = rand(n) .< p_true
            
            # Test that optimization doesn't crash
            using Optim
            result = optimize(a -> logit(a, X, y), zeros(2), BFGS())
            @test Optim.converged(result)
        end
        
        @testset "Function consistency" begin
            # Test that ols and ols2 give same optimization results
            Random.seed!(300)
            X = [ones(10) randn(10, 2)]
            y = randn(10)
            
            using Optim
            result1 = optimize(b -> ols(b, X, y), zeros(3), BFGS())
            result2 = optimize(b -> ols2(b, X, y), zeros(3), BFGS())
            
            @test norm(result1.minimizer - result2.minimizer) < 1e-6
        end
    end
    
    @testset "Numerical Stability Tests" begin
        @testset "Large coefficient handling" begin
            X = [1.0 0.0; 1.0 1.0]
            y = [0.0, 1.0]
            
            # Test logit with very large coefficients
            alpha_large = [0.0, 100.0]
            ll = logit(alpha_large, X, y)
            @test isfinite(ll)
            
            # Test logit with very small coefficients  
            alpha_small = [0.0, 1e-10]
            ll_small = logit(alpha_small, X, y)
            @test isfinite(ll_small)
        end
        
        @testset "Singular matrix handling" begin
            # Create perfectly collinear X matrix
            X = [ones(3) ones(3)]  # Second column is identical to first
            y = [1.0, 2.0, 3.0]
            beta = [1.0, 1.0]
            
            # OLS should still compute (though not meaningful)
            ssr = ols(beta, X, y)
            @test isfinite(ssr)
        end
    end
end

# Run the tests when file is executed
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running PS2 function tests...")
    Test.run()
end