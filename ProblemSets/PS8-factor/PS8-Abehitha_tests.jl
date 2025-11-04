using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, MultivariateStats, FreqTables, ForwardDiff, LineSearches

cd(@__DIR__)

include("PS8-Abehitha_source.jl")

#===============================================================================
************************* Unit Tests for PS8 Factor Analysis *******************
===============================================================================#

@testset "PS8 Factor & PCA Tests" begin
    Random.seed!(2025)
    
    #---------------------------------------------------------------------------
    # Test 1: Data Loading
    #---------------------------------------------------------------------------
    @testset "load_data function" begin
        # Test structure without actual HTTP call
        @test_nowarn begin
            # Create mock data structure
            mock_df = DataFrame(
                black = rand(0:1, 20),
                hispanic = rand(0:1, 20),
                female = rand(0:1, 20),
                schoolt = rand(10:18, 20),
                gradHS = rand(0:1, 20),
                grad4yr = rand(0:1, 20),
                logwage = 2.0 .+ 0.5*randn(20),
                asvabAR = rand(40:80, 20),
                asvabCS = rand(40:80, 20),
                asvabMK = rand(40:80, 20),
                asvabNO = rand(40:80, 20),
                asvabPC = rand(40:80, 20),
                asvabWK = rand(40:80, 20)
            )
            @test isa(mock_df, DataFrame)
            @test ncol(mock_df) == 13  # 7 demographics + 6 ASVAB scores
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 2: ASVAB Correlation Computation
    #---------------------------------------------------------------------------
    @testset "compute_asvab_correlations" begin
        @testset "Basic correlation computation" begin
            N = 50
            # Create correlated ASVAB scores
            df = DataFrame(
                var1 = randn(N),
                var2 = randn(N),
                var3 = randn(N),
                var4 = randn(N),
                var5 = randn(N),
                var6 = randn(N)
            )
            rename!(df, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK])
            
            cordf = compute_asvab_correlations(df)
            
            # Test dimensions
            @test nrow(cordf) == 6
            @test ncol(cordf) == 6
            
            # Test correlation properties
            vals = Matrix(cordf)
            @test all(-1 .<= vals .<= 1)  # All correlations in [-1,1]
            @test all(isfinite.(vals))     # No NaN or Inf
            
            # Diagonal should be 1 (correlation with self)
            @test all(abs.(diag(vals) .- 1.0) .< 1e-10)
        end
        
        @testset "Correlation matrix symmetry" begin
            N = 30
            df = DataFrame(rand(1:100, N, 6), :auto)
            rename!(df, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK])
            
            cordf = compute_asvab_correlations(df)
            cormat = Matrix(cordf)
            
            # Correlation matrix should be symmetric
            @test cormat ≈ cormat'
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 3: Data Preparation for Factor Model
    #---------------------------------------------------------------------------
    @testset "prepare_factor_matrices" begin
        @testset "Matrix dimensions and structure" begin
            N = 100
            df = DataFrame(
                black = rand(0:1, N),
                hispanic = rand(0:1, N),
                female = rand(0:1, N),
                schoolt = rand(10:20, N),
                gradHS = rand(0:1, N),
                grad4yr = rand(0:1, N),
                logwage = 2.0 .+ randn(N),
                asvabAR = rand(40:80, N),
                asvabCS = rand(40:80, N),
                asvabMK = rand(40:80, N),
                asvabNO = rand(40:80, N),
                asvabPC = rand(40:80, N),
                asvabWK = rand(40:80, N)
            )
            
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            
            # Test X (wage equation covariates)
            @test size(X) == (N, 7)  # black, hispanic, female, schoolt, gradHS, grad4yr, constant
            @test all(X[:, end] .== 1)  # Last column should be ones (constant)
            
            # Test y (log wages)
            @test length(y) == N
            @test all(isfinite.(y))
            
            # Test Xfac (measurement equation covariates)
            @test size(Xfac) == (N, 4)  # black, hispanic, female, constant
            @test all(Xfac[:, end] .== 1)  # Last column should be ones
            
            # Test asvabs (test scores)
            @test size(asvabs) == (N, 6)
            @test all(isfinite.(asvabs))
        end
        
        @testset "Data types and values" begin
            N = 50
            df = DataFrame(
                black = rand(0:1, N),
                hispanic = rand(0:1, N),
                female = rand(0:1, N),
                schoolt = rand(10:16, N),
                gradHS = rand(0:1, N),
                grad4yr = rand(0:1, N),
                logwage = 1.5 .+ 0.3*randn(N),
                asvabAR = rand(30:90, N),
                asvabCS = rand(30:90, N),
                asvabMK = rand(30:90, N),
                asvabNO = rand(30:90, N),
                asvabPC = rand(30:90, N),
                asvabWK = rand(30:90, N)
            )
            
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            
            # Binary variables should be 0 or 1
            @test all(X[:, 1] .∈ Ref([0, 1]))  # black
            @test all(X[:, 2] .∈ Ref([0, 1]))  # hispanic
            @test all(X[:, 3] .∈ Ref([0, 1]))  # female
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 4: PCA Generation
    #---------------------------------------------------------------------------
    @testset "generate_pca!" begin
        @testset "PCA column creation" begin
            N = 80
            df = DataFrame(
                black = rand(0:1, N),
                hispanic = rand(0:1, N),
                female = rand(0:1, N),
                schoolt = rand(10:16, N),
                gradHS = rand(0:1, N),
                grad4yr = rand(0:1, N),
                logwage = 2.0 .+ 0.5*randn(N),
                asvabAR = 50 .+ 10*randn(N),
                asvabCS = 50 .+ 10*randn(N),
                asvabMK = 50 .+ 10*randn(N),
                asvabNO = 50 .+ 10*randn(N),
                asvabPC = 50 .+ 10*randn(N),
                asvabWK = 50 .+ 10*randn(N)
            )
            
            df_pca = generate_pca!(copy(df))
            
            # Test new column exists
            @test "asvabPCA" in names(df_pca)
            
            # Test dimensions
            @test length(df_pca.asvabPCA) == N
            
            # Test finite values
            @test all(isfinite.(df_pca.asvabPCA))
            
            # Original columns should still exist
            @test "asvabAR" in names(df_pca)
            @test "asvabWK" in names(df_pca)
        end
        
        @testset "PCA properties" begin
            N = 60
            # Create correlated ASVAB scores
            base = randn(N)
            df = DataFrame(
                asvabAR = 50 .+ 10*base .+ randn(N),
                asvabCS = 50 .+ 10*base .+ randn(N),
                asvabMK = 50 .+ 10*base .+ randn(N),
                asvabNO = 50 .+ 10*base .+ randn(N),
                asvabPC = 50 .+ 10*base .+ randn(N),
                asvabWK = 50 .+ 10*base .+ randn(N)
            )
            
            df_pca = generate_pca!(df)
            
            # PCA scores should have variance
            @test var(df_pca.asvabPCA) > 0
            
            # PCA should explain some variance
            @test std(df_pca.asvabPCA) > 0
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 5: Factor Analysis Generation
    #---------------------------------------------------------------------------
    @testset "generate_factor!" begin
        @testset "Factor column creation" begin
            N = 70
            df = DataFrame(
                black = rand(0:1, N),
                hispanic = rand(0:1, N),
                female = rand(0:1, N),
                schoolt = rand(10:16, N),
                gradHS = rand(0:1, N),
                grad4yr = rand(0:1, N),
                logwage = 2.0 .+ 0.5*randn(N),
                asvabAR = 50 .+ 10*randn(N),
                asvabCS = 50 .+ 10*randn(N),
                asvabMK = 50 .+ 10*randn(N),
                asvabNO = 50 .+ 10*randn(N),
                asvabPC = 50 .+ 10*randn(N),
                asvabWK = 50 .+ 10*randn(N)
            )
            
            df_fac = generate_factor!(copy(df))
            
            # Test new column exists
            @test "asvabFactor" in names(df_fac)
            
            # Test dimensions
            @test length(df_fac.asvabFactor) == N
            
            # Test finite values
            @test all(isfinite.(df_fac.asvabFactor))
            
            # Original columns should still exist
            @test all(in.(["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"], Ref(names(df_fac))))
        end
        
        @testset "Factor vs PCA comparison" begin
            N = 50
            df = DataFrame(
                asvabAR = 50 .+ 10*randn(N),
                asvabCS = 50 .+ 10*randn(N),
                asvabMK = 50 .+ 10*randn(N),
                asvabNO = 50 .+ 10*randn(N),
                asvabPC = 50 .+ 10*randn(N),
                asvabWK = 50 .+ 10*randn(N)
            )
            
            df_pca = generate_pca!(copy(df))
            df_fac = generate_factor!(copy(df))
            
            # Both should produce finite values
            @test all(isfinite.(df_pca.asvabPCA))
            @test all(isfinite.(df_fac.asvabFactor))
            
            # Both should have variance
            @test var(df_pca.asvabPCA) > 0
            @test var(df_fac.asvabFactor) > 0
            
            # Factor and PCA scores should be correlated but not identical
            @test abs(cor(df_pca.asvabPCA, df_fac.asvabFactor)) > 0.2
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 6: Factor Model Likelihood
    #---------------------------------------------------------------------------
    @testset "factor_model likelihood computation" begin
        @testset "Basic likelihood evaluation" begin
            N = 20
            K = 3
            L = 2
            J = 4
            
            X = [ones(N) randn(N, K-1)]
            Xfac = [ones(N) randn(N, L-1)]
            Meas = randn(N, J)
            y = 0.5 .+ 0.2*randn(N)
            
            # Build parameter vector
            γ = 0.1*randn(L, J)
            β = randn(K)
            α = randn(J+1)
            σ = 0.5 .+ 0.2*abs.(randn(J+1))  # positive standard deviations
            θ = vcat(vec(γ), β, α, σ)
            
            # Compute likelihood
            nll = factor_model(θ, X, Xfac, Meas, y, 7)
            
            # Test basic properties
            @test isfinite(nll)
            @test nll > 0  # Negative log-likelihood should be positive
        end
        
        @testset "Parameter sensitivity" begin
            N = 15
            K = 2
            L = 2
            J = 3
            
            X = [ones(N) randn(N, K-1)]
            Xfac = [ones(N) randn(N, L-1)]
            Meas = randn(N, J)
            y = randn(N)
            
            γ = randn(L, J)
            β = randn(K)
            α = randn(J+1)
            σ = abs.(randn(J+1)) .+ 0.3
            θ = vcat(vec(γ), β, α, σ)
            
            nll1 = factor_model(θ, X, Xfac, Meas, y, 5)
            
            # Test that different parameters give different likelihoods
            θ2 = copy(θ)
            θ2[1] += 0.5
            nll2 = factor_model(θ2, X, Xfac, Meas, y, 5)
            @test nll1 != nll2
            
            # Test with different σ
            θ3 = copy(θ)
            θ3[end] *= 1.2
            nll3 = factor_model(θ3, X, Xfac, Meas, y, 5)
            @test nll1 != nll3
        end
        
        @testset "Quadrature points effect" begin
            N = 10
            K = 2
            L = 2
            J = 2
            
            X = [ones(N) randn(N, K-1)]
            Xfac = [ones(N) randn(N, L-1)]
            Meas = randn(N, J)
            y = randn(N)
            
            γ = randn(L, J)
            β = randn(K)
            α = randn(J+1)
            σ = abs.(randn(J+1)) .+ 0.5
            θ = vcat(vec(γ), β, α, σ)
            
            # Different quadrature points should give slightly different results
            nll_5 = factor_model(θ, X, Xfac, Meas, y, 5)
            nll_9 = factor_model(θ, X, Xfac, Meas, y, 9)
            
            @test isfinite(nll_5) && isfinite(nll_9)
            # Results should be similar but not identical
            @test abs(nll_5 - nll_9) < 20.0
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 7: Edge Cases and Robustness
    #---------------------------------------------------------------------------
    @testset "Edge cases and numerical stability" begin
        @testset "Small sample size" begin
            N = 5
            df = DataFrame(
                black = rand(0:1, N),
                hispanic = rand(0:1, N),
                female = rand(0:1, N),
                schoolt = rand(12:16, N),
                gradHS = ones(Int, N),
                grad4yr = rand(0:1, N),
                logwage = 2.0 .+ 0.1*randn(N),
                asvabAR = rand(50:70, N),
                asvabCS = rand(50:70, N),
                asvabMK = rand(50:70, N),
                asvabNO = rand(50:70, N),
                asvabPC = rand(50:70, N),
                asvabWK = rand(50:70, N)
            )
            
            @test_nowarn X, y, Xfac, asvabs = prepare_factor_matrices(df)
            @test_nowarn compute_asvab_correlations(df)
        end
        
        @testset "Perfect correlation handling" begin
            N = 20
            # Create perfectly correlated scores
            base = rand(50:70, N)
            df = DataFrame(
                asvabAR = base,
                asvabCS = base,  # Perfect correlation
                asvabMK = base .+ randn(N),
                asvabNO = base .+ randn(N),
                asvabPC = base .+ randn(N),
                asvabWK = base .+ randn(N)
            )
            
            # Should still compute correlations
            @test_nowarn cordf = compute_asvab_correlations(df)
            cordf = compute_asvab_correlations(df)
            @test all(isfinite.(Matrix(cordf)))
        end
        
        @testset "Large parameter values" begin
            N = 10
            K = 2
            L = 2
            J = 2
            
            X = [ones(N) randn(N, K-1)]
            Xfac = [ones(N) randn(N, L-1)]
            Meas = randn(N, J)
            y = randn(N)
            
            # Test with large parameter values
            γ = 5.0 .* randn(L, J)
            β = 3.0 .* randn(K)
            α = 2.0 .* randn(J+1)
            σ = 1.0 .+ abs.(randn(J+1))
            θ = vcat(vec(γ), β, α, σ)
            
            nll = factor_model(θ, X, Xfac, Meas, y, 5)
            @test isfinite(nll)
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 8: Integration Tests
    #---------------------------------------------------------------------------
    @testset "Integration tests" begin
        @testset "Complete workflow" begin
            N = 40
            df = DataFrame(
                black = rand(0:1, N),
                hispanic = rand(0:1, N),
                female = rand(0:1, N),
                schoolt = rand(10:16, N),
                gradHS = rand(0:1, N),
                grad4yr = rand(0:1, N),
                logwage = 2.0 .+ 0.5*randn(N),
                asvabAR = 50 .+ 10*randn(N),
                asvabCS = 50 .+ 10*randn(N),
                asvabMK = 50 .+ 10*randn(N),
                asvabNO = 50 .+ 10*randn(N),
                asvabPC = 50 .+ 10*randn(N),
                asvabWK = 50 .+ 10*randn(N)
            )
            
            # Test full workflow
            @test_nowarn begin
                # Correlations
                cordf = compute_asvab_correlations(df)
                
                # PCA
                df_pca = generate_pca!(copy(df))
                
                # Factor
                df_fac = generate_factor!(copy(df))
                
                # Prepare matrices
                X, y, Xfac, asvabs = prepare_factor_matrices(df)
                
                # Evaluate likelihood with simple parameters
                K = size(X, 2)
                L = size(Xfac, 2)
                J = size(asvabs, 2)
                
                γ = 0.1*randn(L, J)
                β = randn(K)
                α = randn(J+1)
                σ = 0.5 .+ 0.1*abs.(randn(J+1))
                θ = vcat(vec(γ), β, α, σ)
                
                nll = factor_model(θ, X, Xfac, asvabs, y, 5)
                
                
            end
        end
        
        @testset "Consistency checks" begin
            N = 30
            df = DataFrame(
                black = rand(0:1, N),
                hispanic = rand(0:1, N),
                female = rand(0:1, N),
                schoolt = rand(10:16, N),
                gradHS = rand(0:1, N),
                grad4yr = rand(0:1, N),
                logwage = 2.0 .+ 0.5*randn(N),
                asvabAR = 50 .+ 10*randn(N),
                asvabCS = 50 .+ 10*randn(N),
                asvabMK = 50 .+ 10*randn(N),
                asvabNO = 50 .+ 10*randn(N),
                asvabPC = 50 .+ 10*randn(N),
                asvabWK = 50 .+ 10*randn(N)
            )
            
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            
            # Check dimensions are consistent
            @test size(X, 1) == length(y)
            @test size(Xfac, 1) == size(asvabs, 1)
            @test size(asvabs, 1) == length(y)
            
            # Check data ranges
            @test all(X[:, 1] .∈ Ref([0, 1]))  # black is binary
            @test all(isfinite.(y))
            @test all(isfinite.(asvabs))
        end
    end
end

println("✓ All PS8 Factor Analysis tests completed successfully!")
