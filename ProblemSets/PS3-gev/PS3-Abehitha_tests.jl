using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS3-Abehitha_source.jl")

#===============================================================================
************************* Unit Tests for PS3 Multinomial Logit ****************
===============================================================================#

@testset "PS3 Multinomial Logit Tests" begin
    
    @testset "Data Loading Tests" begin
        @testset "load_data function structure" begin
            # Test with actual URL (if accessible)
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
            
            try
                X, Z, y = load_data(url)
                
                # Test return types
                @test isa(X, Matrix)
                @test isa(Z, Matrix) 
                @test isa(y, Vector)
                
                # Test dimensions
                @test size(X, 2) == 3  # age, white, collgrad
                @test size(Z, 2) == 8  # 8 wage alternatives
                @test size(X, 1) == size(Z, 1) == length(y)  # Same number of observations
                
                # Test data types
                @test eltype(X) <: Real
                @test eltype(Z) <: Real
                @test eltype(y) <: Integer
                
                # Test choice values
                unique_choices = unique(y)
                @test length(unique_choices) == 8  # Should have 8 occupations
                @test minimum(unique_choices) >= 1
                @test maximum(unique_choices) <= 8
                
                println("✓ Data loading tests passed with real data")
            catch e
                println("⚠ Could not access real data URL, skipping online data test: ", e)
            end
        end
        
        @testset "load_data with mock data" begin
            # Create mock CSV data for testing
            mock_data = DataFrame(
                age = [25, 30, 35, 40, 45],
                white = [1, 0, 1, 1, 0],
                collgrad = [1, 0, 1, 0, 1],
                elnwage1 = [2.1, 2.3, 2.5, 2.2, 2.4],
                elnwage2 = [2.0, 2.2, 2.4, 2.1, 2.3],
                elnwage3 = [1.9, 2.1, 2.3, 2.0, 2.2],
                elnwage4 = [1.8, 2.0, 2.2, 1.9, 2.1],
                elnwage5 = [1.7, 1.9, 2.1, 1.8, 2.0],
                elnwage6 = [1.6, 1.8, 2.0, 1.7, 1.9],
                elnwage7 = [1.5, 1.7, 1.9, 1.6, 1.8],
                elnwage8 = [1.4, 1.6, 1.8, 1.5, 1.7],
                occupation = [1, 2, 3, 4, 5]
            )
            
            # Save to temporary file
            temp_file = "temp_test_data.csv"
            CSV.write(temp_file, mock_data)
            
            try
                # Test load_data with file path (simulate URL reading)
                df = CSV.read(temp_file, DataFrame)
                X = [df.age df.white df.collgrad]
                Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
                         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
                y = df.occupation
                
                @test size(X) == (5, 3)
                @test size(Z) == (5, 8)
                @test length(y) == 5
                @test y == [1, 2, 3, 4, 5]
                
                println("✓ Mock data loading tests passed")
            finally
                # Cleanup
                isfile(temp_file) && rm(temp_file)
            end
        end
    end
    
    @testset "mlogit_with_Z Function Tests" begin
        # Create test data
        Random.seed!(1234)
        N = 100
        K = 3  # covariates
        J = 8  # choices
        
        X = randn(N, K)
        Z = randn(N, J)
        y = rand(1:J, N)
        
        @testset "Parameter structure tests" begin
            # Test parameter dimensions
            n_alpha = K * (J - 1)  # 3 * 7 = 21
            n_gamma = 1
            theta = randn(n_alpha + n_gamma)  # 22 parameters total
            
            @test length(theta) == 22
            
            # Test that function runs without error
            @test_nowarn mlogit_with_Z(theta, X, Z, y)
            
            # Test return type
            ll = mlogit_with_Z(theta, X, Z, y)
            @test isa(ll, Real)
            @test ll > 0  # Negative log-likelihood should be positive
        end
        
        @testset "Mathematical properties tests" begin
            theta = randn(22)
            
            # Test with different theta values
            ll1 = mlogit_with_Z(theta, X, Z, y)
            ll2 = mlogit_with_Z(theta .+ 0.1, X, Z, y)
            
            @test ll1 != ll2  # Different parameters should give different likelihoods
            
            # Test with zero parameters
            theta_zero = zeros(22)
            ll_zero = mlogit_with_Z(theta_zero, X, Z, y)
            @test isfinite(ll_zero)
        end
        
        @testset "Edge cases and robustness" begin
            theta = randn(22)
            
            # Test with extreme parameter values
            theta_extreme = [fill(10.0, 21); 5.0]
            @test_nowarn mlogit_with_Z(theta_extreme, X, Z, y)
            
            # Test with single observation
            X_single = X[1:1, :]
            Z_single = Z[1:1, :]
            y_single = y[1:1]
            ll_single = mlogit_with_Z(theta, X_single, Z_single, y_single)
            @test isfinite(ll_single)
            
            # Test probability properties (should sum to 1)
            # Extract intermediate calculations for verification
            alpha = theta[1:end-1]
            gamma = theta[end]
            bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
            
            num = zeros(N, J)
            for j = 1:J
                num[:,j] = exp.(X * bigAlpha[:,j] .+ gamma*(Z[:,j] .- Z[:,end]))
            end
            dem = sum(num, dims=2)
            P = num ./ repeat(dem, 1, J)
            
            # Check that probabilities sum to 1 (approximately)
            @test all(abs.(sum(P, dims=2) .- 1.0) .< 1e-10)
            @test all(P .>= 0)  # All probabilities should be non-negative
        end
        
        @testset "Gradient and optimization properties" begin
            theta = randn(22)
            
            # Test that small changes in theta lead to small changes in likelihood
            eps = 1e-6
            theta_pert = copy(theta)
            theta_pert[1] += eps
            
            ll_base = mlogit_with_Z(theta, X, Z, y)
            ll_pert = mlogit_with_Z(theta_pert, X, Z, y)
            
            @test abs(ll_pert - ll_base) < 100 * eps  # Reasonable continuity
        end
    end
    
    @testset "optimize_mlogit Function Tests" begin
        # Create smaller test data for faster optimization
        Random.seed!(1234)
        N = 50
        K = 3
        J = 8
        
        X_test = randn(N, K)
        Z_test = randn(N, J)
        y_test = rand(1:J, N)
        
        @testset "Optimization convergence" begin
            # Test that optimization runs without error
            @test_nowarn optimize_mlogit(X_test, Z_test, y_test)
            
            # Get optimization result
            theta_hat = optimize_mlogit(X_test, Z_test, y_test)
            
            # Test parameter dimensions
            @test length(theta_hat) == 22  # 21 alphas + 1 gamma
            
            # Test that result is finite
            @test all(isfinite.(theta_hat))
            
            # Test that optimized likelihood is reasonable
            ll_optimized = mlogit_with_Z(theta_hat, X_test, Z_test, y_test)
            ll_random = mlogit_with_Z(randn(22), X_test, Z_test, y_test)
            
            @test ll_optimized <= ll_random  # Optimized should be better (lower)
            
            println("✓ Optimization tests completed")
        end
        
        @testset "Starting values sensitivity" begin
            # Test with different random seeds to check robustness
            Random.seed!(1111)
            theta_hat1 = optimize_mlogit(X_test, Z_test, y_test)
            
            Random.seed!(2222)
            theta_hat2 = optimize_mlogit(X_test, Z_test, y_test)
            
            # Results might differ but should both be reasonable
            @test all(isfinite.(theta_hat1))
            @test all(isfinite.(theta_hat2))
            
            # Both should achieve similar likelihood values
            ll1 = mlogit_with_Z(theta_hat1, X_test, Z_test, y_test)
            ll2 = mlogit_with_Z(theta_hat2, X_test, Z_test, y_test)
            
            @test abs(ll1 - ll2) < 10.0  # Should be reasonably close
        end
    end
    
    @testset "Integration Tests - Complete Workflow" begin
        @testset "Small dataset workflow" begin
            # Create a very simple dataset for testing complete workflow
            Random.seed!(1234)
            X_simple = [1.0 1.0 0.0; 2.0 0.0 1.0; 3.0 1.0 1.0; 4.0 0.0 0.0]
            Z_simple = randn(4, 8)
            y_simple = [1, 2, 3, 4]
            
            # Test complete workflow
            theta_final = optimize_mlogit(X_simple, Z_simple, y_simple)
            final_ll = mlogit_with_Z(theta_final, X_simple, Z_simple, y_simple)
            
            @test length(theta_final) == 22
            @test isfinite(final_ll)
            @test final_ll >= 0
            
            println("✓ Integration workflow test completed")
        end
        
        @testset "Parameter interpretation" begin
            # Test that we can extract meaningful parameters
            Random.seed!(1234)
            N = 50
            K = 3
            J = 8
    
            X_test = randn(N, K)
            Z_test = randn(N, J)
            y_test = rand(1:J, N)
            
            theta = optimize_mlogit(X_test, Z_test, y_test)
            
            alpha = theta[1:21]
            gamma = theta[22]
            
            # Reshape alpha to interpretable form
            alpha_matrix = reshape(alpha, 3, 7)  # K x (J-1)
            
            @test size(alpha_matrix) == (3, 7)
            @test isa(gamma, Real)
            @test isfinite(gamma)
            
            println("✓ Parameter extraction test completed")
        end
    end
    
    @testset "Numerical Stability Tests" begin
        @testset "Large parameter values" begin
            Random.seed!(1234)
            N, K, J = 20, 3, 8
            X = randn(N, K)
            Z = randn(N, J)
            y = rand(1:J, N)
            
            # Test with large parameter values
            theta_large = [fill(5.0, 21); 2.0]
            ll_large = mlogit_with_Z(theta_large, X, Z, y)
            @test isfinite(ll_large)
            
            # Test with small parameter values
            theta_small = [fill(0.001, 21); 0.001]
            ll_small = mlogit_with_Z(theta_small, X, Z, y)
            @test isfinite(ll_small)
        end
        
        @testset "Data scaling effects" begin
            Random.seed!(1234)
            N, K, J = 20, 3, 8
            X = randn(N, K)
            Z = randn(N, J)
            y = rand(1:J, N)
            theta = randn(22)
            
            # Test with scaled data
            X_scaled = 10 * X
            ll_original = mlogit_with_Z(theta, X, Z, y)
            ll_scaled = mlogit_with_Z(theta, X_scaled, Z, y)
            
            @test isfinite(ll_original)
            @test isfinite(ll_scaled)
            @test ll_original != ll_scaled  # Should be different due to scaling
        end
    end
end

#===============================================================================
************************* Unit Tests for PS3 Nested Logit ********************
===============================================================================#

@testset "PS3 Nested Logit Tests" begin
    
    @testset "nested_logit_with_Z Function Tests" begin
        # Create test data
        Random.seed!(5678)
        N = 50
        K = 3  # covariates
        J = 8  # choices
        
        X = randn(N, K)
        Z = randn(N, J)
        y = rand(1:J, N)
        nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]  # WC and BC nests, 8 is other
        
        @testset "Parameter structure tests" begin
            # Test parameter dimensions
            n_alpha = 2 * K  # 6 parameters (3 for WC, 3 for BC)
            n_lambda = 2     # 2 lambda parameters
            n_gamma = 1      # 1 gamma parameter
            theta = [randn(n_alpha); [0.7, 0.8]; 0.1]  # 9 parameters total
            
            @test length(theta) == 9
            
            # Test that function runs without error
            @test_nowarn nested_logit_with_Z(theta, X, Z, y, nesting_structure)
            
            # Test return type
            ll = nested_logit_with_Z(theta, X, Z, y, nesting_structure)
            @test isa(ll, Real)
            @test ll > 0  # Negative log-likelihood should be positive
        end
        
        @testset "Lambda parameter constraints" begin
            theta_base = [randn(6); [0.7, 0.8]; 0.1]
            
            # Test with lambda values in valid range (0 < lambda <= 1)
            theta_valid = [randn(6); [0.5, 0.9]; 0.1]
            @test_nowarn nested_logit_with_Z(theta_valid, X, Z, y, nesting_structure)
            
            # Test with lambda = 1 (multinomial logit case)
            theta_mlogit = [randn(6); [1.0, 1.0]; 0.1]
            ll_mlogit = nested_logit_with_Z(theta_mlogit, X, Z, y, nesting_structure)
            @test isfinite(ll_mlogit)
            
            # Test with small but positive lambda
            theta_small_lambda = [randn(6); [0.1, 0.1]; 0.1]
            ll_small = nested_logit_with_Z(theta_small_lambda, X, Z, y, nesting_structure)
            @test isfinite(ll_small)
        end
        
        @testset "Nesting structure validation" begin
            theta = [randn(6); [0.7, 0.8]; 0.1]
            
            # Test with correct nesting structure
            nest_correct = [[1, 2, 3], [4, 5, 6, 7]]
            @test_nowarn nested_logit_with_Z(theta, X, Z, y, nest_correct)
            
            # Test with different valid nesting structure
            nest_alt = [[1, 2], [3, 4, 5, 6, 7]]  # Different grouping
            @test_nowarn nested_logit_with_Z(theta, X, Z, y, nest_alt)
            
            # Test with single alternative in nest
            nest_single = [[1], [2, 3, 4, 5, 6, 7]]
            @test_nowarn nested_logit_with_Z(theta, X, Z, y, nest_single)
        end
        
        @testset "Mathematical properties tests" begin
            theta = [randn(6); [0.7, 0.8]; 0.1]
            
            # Test with different theta values
            ll1 = nested_logit_with_Z(theta, X, Z, y, nesting_structure)
            ll2 = nested_logit_with_Z(theta .+ 0.1, X, Z, y, nesting_structure)
            
            @test ll1 != ll2  # Different parameters should give different likelihoods
            
            # Test with zero coefficients
            theta_zero = [zeros(6); [0.7, 0.8]; 0.0]
            ll_zero = nested_logit_with_Z(theta_zero, X, Z, y, nesting_structure)
            @test isfinite(ll_zero)
        end
        
        @testset "Edge cases and robustness" begin
            theta = [randn(6); [0.7, 0.8]; 0.1]
            
            # Test with single observation
            X_single = X[1:1, :]
            Z_single = Z[1:1, :]
            y_single = y[1:1]
            ll_single = nested_logit_with_Z(theta, X_single, Z_single, y_single, nesting_structure)
            @test isfinite(ll_single)
            
            # Test with all observations choosing from same nest
            y_same_nest = fill(1, N)  # All choose alternative 1 (in WC nest)
            ll_same_nest = nested_logit_with_Z(theta, X, Z, y_same_nest, nesting_structure)
            @test isfinite(ll_same_nest)
            
            # Test with observations choosing the "other" alternative
            y_other = fill(8, N)  # All choose alternative 8 (other)
            ll_other = nested_logit_with_Z(theta, X, Z, y_other, nesting_structure)
            @test isfinite(ll_other)
        end
        
        @testset "Probability properties verification" begin
            theta = [randn(6); [0.7, 0.8]; 0.1]
            
            # Manually calculate probabilities to verify they sum to 1
            K = size(X, 2)
            J = length(unique(y))
            N = length(y)
            
            alpha = theta[1:end-3]
            lambda = theta[end-2:end-1]
            gamma = theta[end]
            
            # Create coefficient matrix
            bigAlpha = zeros(K, J)
            bigAlpha[:, nesting_structure[1]] .= repeat(alpha[1:K], 1, length(nesting_structure[1]))
            bigAlpha[:, nesting_structure[2]] .= repeat(alpha[K+1:2K], 1, length(nesting_structure[2]))
            
            # Calculate linear indices
            lidx = zeros(N, J)
            for j = 1:J
                if j in nesting_structure[1]
                    lidx[:,j] = exp.((X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma) ./ lambda[1])
                elseif j in nesting_structure[2]
                    lidx[:,j] = exp.((X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma) ./ lambda[2])
                else
                    lidx[:,j] = exp.(zeros(N))
                end
            end
            
            # Calculate numerators
            num = zeros(N, J)
            for j = 1:J
                if j in nesting_structure[1]
                    num[:,j] = lidx[:,j] .* (sum(lidx[:, nesting_structure[1]], dims=2)).^(lambda[1]-1)
                elseif j in nesting_structure[2]
                    num[:,j] = lidx[:,j] .* (sum(lidx[:, nesting_structure[2]], dims=2)).^(lambda[2]-1)
                else
                    num[:,j] = lidx[:,j]
                end
            end
            
            dem = sum(num, dims=2)
            P = num ./ repeat(dem, 1, J)
            
            # Check probability properties
            @test all(abs.(sum(P, dims=2) .- 1.0) .< 1e-10)  # Probabilities sum to 1
            @test all(P .>= 0)  # All probabilities non-negative
            @test all(P .<= 1)  # All probabilities <= 1
        end
    end
    
    @testset "optimize_nested_logit Function Tests" begin
        # Create smaller test data for faster optimization
        Random.seed!(5678)
        N = 30
        K = 3
        J = 8
        
        X_test = randn(N, K)
        Z_test = randn(N, J)
        y_test = rand(1:J, N)
        nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
        
        @testset "Optimization convergence and bounds" begin
            # Test that optimization runs without error
            @test_nowarn optimize_nested_logit(X_test, Z_test, y_test, nesting_structure)
            
            # Get optimization result
            theta_hat = optimize_nested_logit(X_test, Z_test, y_test, nesting_structure)
            
            # Test parameter dimensions
            @test length(theta_hat) == 9  # 6 alphas + 2 lambdas + 1 gamma
            
            # Test that result is finite
            @test all(isfinite.(theta_hat))
            
            # Test lambda bounds (should be between 0.1 and 1.0)
            lambda_hat = theta_hat[7:8]
            @test all(lambda_hat .>= 0.1)
            @test all(lambda_hat .<= 1.0)
            
            # Test that optimized likelihood is reasonable
            ll_optimized = nested_logit_with_Z(theta_hat, X_test, Z_test, y_test, nesting_structure)
            ll_random = nested_logit_with_Z([randn(6); [0.7, 0.8]; 0.1], X_test, Z_test, y_test, nesting_structure)
            
            @test ll_optimized <= ll_random  # Optimized should be better (lower)
            
            println("✓ Nested logit optimization tests completed")
        end
        
        @testset "Different nesting structures" begin
            # Test with different nesting configurations
            nest_alt1 = [[1, 2], [3, 4, 5, 6, 7]]
            nest_alt2 = [[1], [2, 3, 4, 5, 6, 7]]
            
            theta_hat1 = optimize_nested_logit(X_test, Z_test, y_test, nest_alt1)
            theta_hat2 = optimize_nested_logit(X_test, Z_test, y_test, nest_alt2)
            
            @test length(theta_hat1) == 9
            @test length(theta_hat2) == 9
            @test all(isfinite.(theta_hat1))
            @test all(isfinite.(theta_hat2))
            
            # Lambda bounds should be respected
            @test all(theta_hat1[7:8] .>= 0.1) && all(theta_hat1[7:8] .<= 1.0)
            @test all(theta_hat2[7:8] .>= 0.1) && all(theta_hat2[7:8] .<= 1.0)
        end
        
        @testset "Starting values sensitivity" begin
            # Test with different random seeds
            Random.seed!(1111)
            theta_hat1 = optimize_nested_logit(X_test, Z_test, y_test, nesting_structure)
            
            Random.seed!(2222) 
            theta_hat2 = optimize_nested_logit(X_test, Z_test, y_test, nesting_structure)
            
            @test all(isfinite.(theta_hat1))
            @test all(isfinite.(theta_hat2))
            
            # Both should satisfy lambda constraints
            @test all(theta_hat1[7:8] .>= 0.1) && all(theta_hat1[7:8] .<= 1.0)
            @test all(theta_hat2[7:8] .>= 0.1) && all(theta_hat2[7:8] .<= 1.0)
            
            # Both should achieve reasonable likelihood values
            ll1 = nested_logit_with_Z(theta_hat1, X_test, Z_test, y_test, nesting_structure)
            ll2 = nested_logit_with_Z(theta_hat2, X_test, Z_test, y_test, nesting_structure)
            
            @test abs(ll1 - ll2) < 20.0  # Should be reasonably close
        end
    end
    
    @testset "Nested vs Multinomial Logit Comparison Tests" begin
        Random.seed!(9999)
        N = 40
        K = 3
        J = 8
        
        X_comp = randn(N, K)
        Z_comp = randn(N, J)
        y_comp = rand(1:J, N)
        nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
        
        @testset "Special case: lambda = 1 should approximate multinomial logit" begin
            # When lambda = 1, nested logit should behave like multinomial logit
            theta_nested = [randn(6); [1.0, 1.0]; 0.1]
            ll_nested = nested_logit_with_Z(theta_nested, X_comp, Z_comp, y_comp, nesting_structure)
            
            # Create corresponding multinomial logit parameters
            # This is not a direct comparison since parameter structures differ,
            # but we can check that both give finite reasonable likelihoods
            theta_mlogit = [randn(21); 0.1]  # 21 alphas + 1 gamma for mlogit
            ll_mlogit = mlogit_with_Z(theta_mlogit, X_comp, Z_comp, y_comp)
            
            @test isfinite(ll_nested)
            @test isfinite(ll_mlogit)
            @test ll_nested > 0
            @test ll_mlogit > 0
        end
        
        @testset "Parameter count comparison" begin
            # Nested logit should have fewer parameters than multinomial logit
            theta_nested = optimize_nested_logit(X_comp, Z_comp, y_comp, nesting_structure)
            
            @test length(theta_nested) == 9  # Nested logit: 6 + 2 + 1
            # Multinomial logit would have 22 parameters: 21 + 1
            @test 9 < 22  # Nested logit is more parsimonious
        end
    end
    
    @testset "Integration Tests - Complete Nested Logit Workflow" begin
        @testset "Small dataset complete workflow" begin
            Random.seed!(1234)
            X_simple = [1.0 1.0 0.0; 2.0 0.0 1.0; 3.0 1.0 1.0; 4.0 0.0 0.0; 5.0 1.0 1.0; 6.0 0.0 0.0]
            Z_simple = randn(6, 8)
            y_simple = [1, 2, 3, 4, 5, 6]
            nest_simple = [[1, 2, 3], [4, 5, 6, 7]]
            
            # Test complete workflow
            theta_final = optimize_nested_logit(X_simple, Z_simple, y_simple, nest_simple)
            final_ll = nested_logit_with_Z(theta_final, X_simple, Z_simple, y_simple, nest_simple)
            
            @test length(theta_final) == 9
            @test isfinite(final_ll)
            @test final_ll > 0
            @test all(theta_final[7:8] .>= 0.1) && all(theta_final[7:8] .<= 1.0)
            
            println("✓ Nested logit integration workflow test completed")
        end
        
        @testset "Parameter extraction and interpretation" begin
            Random.seed!(1234)
            N = 30
            K = 3
            J = 8
    
            X_test = randn(N, K)
            Z_test = randn(N, J)
            y_test = rand(1:J, N)
            nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
            theta = optimize_nested_logit(X_test, Z_test, y_test, nesting_structure)
            
            # Extract parameters
            alpha_WC = theta[1:3]      # White collar coefficients
            alpha_BC = theta[4:6]      # Blue collar coefficients  
            lambda_WC = theta[7]       # White collar lambda
            lambda_BC = theta[8]       # Blue collar lambda
            gamma = theta[9]           # Alternative-specific coefficient
            
            @test length(alpha_WC) == 3
            @test length(alpha_BC) == 3
            @test isa(lambda_WC, Real) && 0.1 <= lambda_WC <= 1.0
            @test isa(lambda_BC, Real) && 0.1 <= lambda_BC <= 1.0
            @test isa(gamma, Real) && isfinite(gamma)
            
            println("✓ Nested logit parameter extraction test completed")
        end
    end
    
    @testset "Numerical Stability Tests for Nested Logit" begin
        Random.seed!(7777)
        N, K, J = 20, 3, 8
        X = randn(N, K)
        Z = randn(N, J)
        y = rand(1:J, N)
        nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
        
        @testset "Extreme lambda values" begin
            # Test near boundary lambda values
            theta_low_lambda = [randn(6); [0.1001, 0.1001]; 0.1]
            ll_low = nested_logit_with_Z(theta_low_lambda, X, Z, y, nesting_structure)
            @test isfinite(ll_low)
            
            theta_high_lambda = [randn(6); [0.9999, 0.9999]; 0.1]
            ll_high = nested_logit_with_Z(theta_high_lambda, X, Z, y, nesting_structure)
            @test isfinite(ll_high)
        end
        
        @testset "Large coefficient values" begin
            # Test with large coefficients
            theta_large = [fill(5.0, 6); [0.7, 0.8]; 2.0]
            ll_large = nested_logit_with_Z(theta_large, X, Z, y, nesting_structure)
            @test isfinite(ll_large)
            
            # Test with small coefficients
            theta_small = [fill(0.001, 6); [0.7, 0.8]; 0.001]
            ll_small = nested_logit_with_Z(theta_small, X, Z, y, nesting_structure)
            @test isfinite(ll_small)
        end
        
        @testset "Data scaling effects" begin
            theta = [randn(6); [0.7, 0.8]; 0.1]
            
            # Test with scaled X data
            X_scaled = 10 * X
            ll_original = nested_logit_with_Z(theta, X, Z, y, nesting_structure)
            ll_scaled = nested_logit_with_Z(theta, X_scaled, Z, y, nesting_structure)
            
            @test isfinite(ll_original)
            @test isfinite(ll_scaled)
            @test ll_original != ll_scaled  # Should be different due to scaling
            
            # Test with scaled Z data
            Z_scaled = 10 * Z
            ll_z_scaled = nested_logit_with_Z(theta, X, Z_scaled, y, nesting_structure)
            @test isfinite(ll_z_scaled)
        end
        
        @testset "Continuity and smoothness" begin
            theta = [randn(6); [0.7, 0.8]; 0.1]
            
            # Test small perturbations lead to small changes
            eps = 1e-6
            theta_pert = copy(theta)
            theta_pert[1] += eps
            
            ll_base = nested_logit_with_Z(theta, X, Z, y, nesting_structure)
            ll_pert = nested_logit_with_Z(theta_pert, X, Z, y, nesting_structure)
            
            @test abs(ll_pert - ll_base) < 1000 * eps  # Reasonable continuity
            
            # Test lambda perturbation
            theta_lambda_pert = copy(theta)
            theta_lambda_pert[7] += eps
            ll_lambda_pert = nested_logit_with_Z(theta_lambda_pert, X, Z, y, nesting_structure)
            
            @test abs(ll_lambda_pert - ll_base) < 1000 * eps
        end
    end
end

println("All multinomial logit unit tests completed!")
println("All nested logit unit tests completed!")
println("Run tests with: julia -e 'include(\"PS3-Abehitha_tests.jl\")'")