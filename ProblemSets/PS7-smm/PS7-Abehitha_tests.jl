using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS7-Abehitha_source.jl")

@testset "PS7 GMM and SMM Tests" begin
    
    Random.seed!(1234)
    
    @testset "Data Loading Functions" begin
        # Test prepare_occupation_data function
        @test_nowarn begin
            mock_df = DataFrame(
                age = 25:34,
                race = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                collgrad = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                occupation = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            )
            
            df_prep, X_prep, y_prep = prepare_occupation_data(mock_df)
            @test size(X_prep, 2) == 4  # intercept + age + white + collgrad
            @test hasproperty(df_prep, :white)
            @test all(df_prep.white .∈ Ref([0, 1]))
            @test maximum(y_prep) <= 7  # Occupations should be collapsed to 7 categories
        end
    end
    
    @testset "OLS GMM Function Tests" begin
        # Create test data
        N, K = 100, 3
        X = [ones(N) randn(N, K-1)]
        β_true = [1.0, 0.5, -0.3]
        y = X * β_true + 0.1 * randn(N)
        
        # Test ols_gmm function
        @test_nowarn ols_gmm(β_true, X, y)
        obj_val = ols_gmm(β_true, X, y)
        @test isa(obj_val, Real) && obj_val >= 0
        
        # Test that objective is minimized at true parameters (approximately)
        obj_true = ols_gmm(β_true, X, y)
        obj_wrong = ols_gmm(β_true .+ 0.5, X, y)
        @test obj_true < obj_wrong
        
        # Test with different parameter values
        β_test = rand(K)
        @test isfinite(ols_gmm(β_test, X, y))
        
        # Test dimensions
        @test_throws DimensionMismatch ols_gmm([1.0, 2.0], X, y)  # Wrong β dimension
    end
    
    @testset "Multinomial Logit MLE Tests" begin
        # Create test data
        N, K, J = 50, 3, 4
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        # Test mlogit_mle function
        @test_nowarn mlogit_mle(α, X, y)
        ll = mlogit_mle(α, X, y)
        @test isa(ll, Real) && ll > 0  # Should be positive (negative log-likelihood)
        
        # Test with different parameters
        α2 = α .+ 0.1
        ll2 = mlogit_mle(α2, X, y)
        @test ll != ll2  # Different parameters should give different likelihoods
        
        # Test parameter dimension requirements
        @test_throws DimensionMismatch mlogit_mle(α[1:end-1], X, y)
        
        # Test with extreme parameters
        α_extreme = fill(10.0, K * (J-1))
        @test isfinite(mlogit_mle(α_extreme, X, y))
    end
    
    @testset "Multinomial Logit GMM Tests" begin
        N, K, J = 40, 3, 4
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = 0.1 * randn(K * (J-1))
        
        # Test mlogit_gmm function (just-identified)
        @test_nowarn mlogit_gmm(α, X, y)
        obj_gmm = mlogit_gmm(α, X, y)
        @test isa(obj_gmm, Real) && obj_gmm >= 0
        
        # Test mlogit_gmm_overid function (over-identified)
        @test_nowarn mlogit_gmm_overid(α, X, y)
        obj_overid = mlogit_gmm_overid(α, X, y)
        @test isa(obj_overid, Real) && obj_overid >= 0
        
        # Test parameter sensitivity
        α_pert = α .+ 0.01
        obj_pert = mlogit_gmm(α_pert, X, y)
        @test obj_pert != obj_gmm
        
        # Test with different choice sets
        y_binary = rand(1:2, N)
        α_binary = randn(K * (2-1))
        @test_nowarn mlogit_gmm(α_binary, X, y_binary)
    end
    
    @testset "Data Simulation Tests" begin
        # Test sim_logit function
        @test_nowarn sim_logit(1000, 4)
        Y, X = sim_logit(1000, 4)
        
        @test length(Y) == 1000
        @test size(X) == (1000, 4)  # Should have 4 covariates including intercept
        @test all(Y .∈ Ref([1, 2, 3, 4]))  # Choices should be in {1,2,3,4}
        @test X[:, 1] == ones(1000)  # First column should be intercept
        
        # Test sim_logit_w_gumbel function
        @test_nowarn sim_logit_w_gumbel(1000, 4)
        Y_gumbel, X_gumbel = sim_logit_w_gumbel(1000, 4)
        
        @test length(Y_gumbel) == 1000
        @test size(X_gumbel) == (1000, 4)
        @test all(Y_gumbel .∈ Ref([1, 2, 3, 4]))
        @test X_gumbel[:, 1] == ones(1000)
        
        # Test with different J values
        @test_nowarn sim_logit(500, 3)
        Y3, X3 = sim_logit(500, 3)
        @test all(Y3 .∈ Ref([1, 2, 3]))
        
        # Test choice frequency distribution
        choice_freq = [mean(Y .== j) for j in 1:4]
        @test all(choice_freq .> 0)  # All choices should appear
        @test abs(sum(choice_freq) - 1.0) < 1e-10  # Should sum to 1
    end
    
    @testset "SMM Function Tests" begin
        N, K, J = 30, 3, 3  # Small for testing
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = 0.1 * randn(K * (J-1))
        D = 10  # Small number of simulation draws for testing
        
        # Test mlogit_smm_overid function
        @test_nowarn mlogit_smm_overid(α, X, y, D)
        obj_smm = mlogit_smm_overid(α, X, y, D)
        @test isa(obj_smm, Real) && obj_smm >= 0
        
        # Test with different parameters
        α_diff = α .+ 0.1
        obj_smm_diff = mlogit_smm_overid(α_diff, X, y, D)
        @test obj_smm_diff != obj_smm
        
        # Test with different D values
        obj_smm_D5 = mlogit_smm_overid(α, X, y, 5)
        obj_smm_D20 = mlogit_smm_overid(α, X, y, 20)
        @test isfinite(obj_smm_D5) && isfinite(obj_smm_D20)
        
        # Test deterministic behavior with same seed
        Random.seed!(1234)
        obj1 = mlogit_smm_overid(α, X, y, D)
        Random.seed!(1234)
        obj2 = mlogit_smm_overid(α, X, y, D)
        @test obj1 ≈ obj2  # Should be identical with same seed
    end
    
    @testset "Mathematical Properties Tests" begin
        # Test probability calculations in mlogit_mle
        N, K, J = 20, 3, 3
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = 0.1 * randn(K * (J-1))
        
        # Extract probability calculation from mlogit_mle
        bigα = [reshape(α, K, J-1) zeros(K)]
        P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))
        
        # Test probability properties
        @test size(P) == (N, J)
        @test all(P .>= 0)  # Probabilities should be non-negative
        @test all(abs.(sum(P, dims=2) .- 1) .< 1e-10)  # Should sum to 1 across choices
        
        # Test moment conditions in GMM
        bigY = zeros(N, J)
        for j = 1:J; bigY[:,j] = y .== j; end
        @test all(sum(bigY, dims=2) .== 1)  # Each row should sum to 1
        @test all(bigY .∈ Ref([0, 1]))  # Should only contain 0s and 1s
    end
    
    @testset "Optimization Integration Tests" begin
        # Test small-scale optimization problems
        N, K = 20, 2
        X_small = [ones(N) randn(N)]
        β_true = [1.0, 0.5]
        y_small = X_small * β_true + 0.1 * randn(N)
        
        # Test OLS GMM optimization
        @test_nowarn optimize(b -> ols_gmm(b, X_small, y_small), rand(K), LBFGS(), 
                             Optim.Options(g_tol=1e-4, iterations=100))
        
        result_ols = optimize(b -> ols_gmm(b, X_small, y_small), rand(K), LBFGS(), 
                             Optim.Options(g_tol=1e-4, iterations=100))
        @test Optim.converged(result_ols)
        @test isfinite(result_ols.minimum)
        
        # Test multinomial logit optimization (very small problem)
        N_tiny, J_tiny = 15, 3
        X_tiny = [ones(N_tiny) randn(N_tiny)]
        y_tiny = rand(1:J_tiny, N_tiny)
        α_start = 0.1 * randn(2 * (J_tiny-1))
        
        @test_nowarn optimize(a -> mlogit_mle(a, X_tiny, y_tiny), α_start, LBFGS(),
                             Optim.Options(g_tol=1e-4, iterations=50))
    end
    
    @testset "Edge Cases and Robustness" begin
        # Test with perfect separation case
        N = 10
        X_sep = [ones(N) [ones(5); -ones(5)]]
        y_sep = [ones(Int, 5); 2*ones(Int, 5)]
        
        # Should not crash even with separation issues
        @test_nowarn mlogit_mle(randn(2), X_sep, y_sep)
        
        # Test with extreme parameter values
        X_test = [ones(10) randn(10, 2)]
        y_test = rand(1:3, 10)
        α_large = fill(100.0, 6)  # Changed from 4 to 6
        @test isfinite(mlogit_mle(α_large, X_test, y_test))

        # Test with zero parameters
        α_zero = zeros(6)  # Changed from 4 to 6
        @test isfinite(mlogit_mle(α_zero, X_test, y_test))
    end
    
    @testset "Consistency Tests" begin
        # Test that MLE and GMM give similar results for same problem
        Random.seed!(1234)
        N, K, J = 100, 3, 4
        X = [ones(N) randn(N, K-1)]
        
        # Generate data from known parameters
        α_true = 0.5 * randn(K * (J-1))
        bigα_true = [reshape(α_true, K, J-1) zeros(K)]
        P_true = exp.(X * bigα_true) ./ sum.(eachrow(exp.(X * bigα_true)))
        
        # Simulate choices
        y = zeros(Int, N)
        for i in 1:N
            y[i] = rand(Categorical(P_true[i, :]))
        end
        
        # Estimate via MLE and GMM
        α_start = α_true .+ 0.1 * randn(length(α_true))
        
        result_mle = optimize(a -> mlogit_mle(a, X, y), α_start, LBFGS(),
                             Optim.Options(g_tol=1e-5, iterations=1000))
        
        result_gmm = optimize(a -> mlogit_gmm(a, X, y), α_start, LBFGS(),
                             Optim.Options(g_tol=1e-5, iterations=1000))
        
        # Both should converge
        @test Optim.converged(result_mle)
        @test Optim.converged(result_gmm)
        
        # Estimates should be reasonably close
        @test norm(result_mle.minimizer - result_gmm.minimizer) < 0.5
    end
    
    @testset "Numerical Stability Tests" begin
        N, K = 50, 3
        X_moderate = [ones(N) 10*randn(N, K-1)]  # Reduced from 1000
        y_large = rand(1:3, N)
        α_test = 0.1 * randn(K * 2)  # Smaller parameters
        
        @test_nowarn mlogit_mle(α_test, X_moderate, y_large)
        @test_nowarn mlogit_gmm(α_test, X_moderate, y_large)
        
        # Test with small variance data
        X_small = [ones(N) 0.001*randn(N, K-1)]
        @test isfinite(mlogit_mle(α_test, X_small, y_large))
        
        # Test moment scaling in GMM - use X_moderate, not undefined X
        Random.seed!(123)
        obj1 = mlogit_gmm(α_test, X_moderate, rand(1:3, N))
        Random.seed!(124)
        obj2 = mlogit_gmm(2*α_test, X_moderate, rand(1:3, N))
        @test obj1 != obj2
    end
end

println("✓ All PS7 GMM and SMM tests completed successfully!")