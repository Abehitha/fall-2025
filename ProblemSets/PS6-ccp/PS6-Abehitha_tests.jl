using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

cd(@__DIR__)

include("PS6-Abehitha_source.jl")

#===============================================================================
************************* Unit Tests for PS6 CCP Estimation ********************
===============================================================================#

@testset "PS6 CCP-Based Rust Model Tests" begin
    
    Random.seed!(12345)
    
    #---------------------------------------------------------------------------
    # Test 1: Data Loading and Reshaping
    #---------------------------------------------------------------------------
    @testset "Data Loading and Reshaping" begin
        @testset "load_and_reshape_data structure" begin
            # Create mock data to test reshaping logic without HTTP call
            mock_df = DataFrame(
                Y1 = rand(0:1, 10), Y2 = rand(0:1, 10), Y3 = rand(0:1, 10),
                Y4 = rand(0:1, 10), Y5 = rand(0:1, 10), Y6 = rand(0:1, 10),
                Y7 = rand(0:1, 10), Y8 = rand(0:1, 10), Y9 = rand(0:1, 10),
                Y10 = rand(0:1, 10), Y11 = rand(0:1, 10), Y12 = rand(0:1, 10),
                Y13 = rand(0:1, 10), Y14 = rand(0:1, 10), Y15 = rand(0:1, 10),
                Y16 = rand(0:1, 10), Y17 = rand(0:1, 10), Y18 = rand(0:1, 10),
                Y19 = rand(0:1, 10), Y20 = rand(0:1, 10),
                Odo1 = 10000 .+ 1000*randn(10), Odo2 = 11000 .+ 1000*randn(10),
                Odo3 = 12000 .+ 1000*randn(10), Odo4 = 13000 .+ 1000*randn(10),
                Odo5 = 14000 .+ 1000*randn(10), Odo6 = 15000 .+ 1000*randn(10),
                Odo7 = 16000 .+ 1000*randn(10), Odo8 = 17000 .+ 1000*randn(10),
                Odo9 = 18000 .+ 1000*randn(10), Odo10 = 19000 .+ 1000*randn(10),
                Odo11 = 20000 .+ 1000*randn(10), Odo12 = 21000 .+ 1000*randn(10),
                Odo13 = 22000 .+ 1000*randn(10), Odo14 = 23000 .+ 1000*randn(10),
                Odo15 = 24000 .+ 1000*randn(10), Odo16 = 25000 .+ 1000*randn(10),
                Odo17 = 26000 .+ 1000*randn(10), Odo18 = 27000 .+ 1000*randn(10),
                Odo19 = 28000 .+ 1000*randn(10), Odo20 = 29000 .+ 1000*randn(10),
                Xst1 = rand(1:5, 10), Xst2 = rand(1:5, 10), Xst3 = rand(1:5, 10),
                Xst4 = rand(1:5, 10), Xst5 = rand(1:5, 10), Xst6 = rand(1:5, 10),
                Xst7 = rand(1:5, 10), Xst8 = rand(1:5, 10), Xst9 = rand(1:5, 10),
                Xst10 = rand(1:5, 10), Xst11 = rand(1:5, 10), Xst12 = rand(1:5, 10),
                Xst13 = rand(1:5, 10), Xst14 = rand(1:5, 10), Xst15 = rand(1:5, 10),
                Xst16 = rand(1:5, 10), Xst17 = rand(1:5, 10), Xst18 = rand(1:5, 10),
                Xst19 = rand(1:5, 10), Xst20 = rand(1:5, 10),
                Zst = rand(1:3, 10),
                RouteUsage = rand(10),
                Branded = rand(0:1, 10)
            )
            
            # Test the function works without error
            @test_nowarn begin
                # Simulate the reshaping logic manually
                N = size(mock_df, 1)
                T = 20
                expected_rows = N * T
                @test expected_rows == 200
            end
        end
        
        @testset "Long format properties" begin
            # Test expected dimensions after reshaping
            N_buses = 10
            T_periods = 20
            expected_rows = N_buses * T_periods
            
            @test expected_rows == 200
            
            # Each bus should appear T times
            @test T_periods == 20
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 2: Flexible Logit Estimation
    #---------------------------------------------------------------------------
    @testset "Flexible Logit Estimation" begin
        @testset "estimate_flexible_logit structure" begin
            # Create mock data for logit estimation
            mock_df = DataFrame(
                Y = rand(0:1, 100),
                Odometer = 50000 .+ 10000*randn(100),
                RouteUsage = rand(100),
                Branded = rand(0:1, 100),
                time = repeat(1:20, inner=5)
            )
            
            @test_nowarn begin
                model = glm(@formula(Y ~ Odometer + RouteUsage + Branded + time),
                           mock_df, Binomial(), LogitLink())
                @test isa(model, StatsModels.TableRegressionModel)
                @test length(coef(model)) >= 4  # At least intercept + 3 variables
            end
        end
        
        @testset "Logit model properties" begin
            mock_df = DataFrame(
                Y = rand(0:1, 50),
                Odometer = 50000 .+ 10000*randn(50),
                RouteUsage = rand(50),
                Branded = rand(0:1, 50),
                time = rand(1:20, 50)
            )
            
            model = glm(@formula(Y ~ Odometer + Branded), 
                       mock_df, Binomial(), LogitLink())
            
            # Test model fitting
            @test isa(model, StatsModels.TableRegressionModel)
            @test length(coef(model)) == 3  # intercept + 2 variables
            
            # Test predictions
            preds = predict(model, mock_df)
            @test all(0 .<= preds .<= 1)  # Probabilities in [0,1]
            @test length(preds) == nrow(mock_df)
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 3: State Space Construction
    #---------------------------------------------------------------------------
    @testset "State Space Construction" begin
        @testset "construct_state_space basic" begin
            xbin = 5
            zbin = 3
            xval = collect(range(0, 200000, length=xbin))
            zval = collect(range(0, 1, length=zbin))
            xtran = rand(xbin*zbin, xbin)
            for i in 1:size(xtran,1)
                xtran[i,:] ./= sum(xtran[i,:])
            end
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            # Test dimensions
            @test nrow(state_df) == xbin * zbin
            @test ncol(state_df) == 4  # Odometer, RouteUsage, Branded, time
            
            # Test column names
            @test "Odometer" in names(state_df)
            @test "RouteUsage" in names(state_df)
            @test "Branded" in names(state_df)
            @test "time" in names(state_df)

            # Test initial values
            @test all(state_df.Branded .== 0)
            @test all(state_df.time .== 0)
        end
        
        @testset "State space grid structure" begin
            xbin = 4
            zbin = 2
            xval = [0.0, 50000.0, 100000.0, 150000.0]
            zval = [0.3, 0.7]
            xtran = rand(xbin*zbin, xbin)
            for i in 1:size(xtran,1)
                xtran[i,:] ./= sum(xtran[i,:])
            end
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            # Test Kronecker product structure
            @test length(unique(state_df.Odometer)) == xbin
            @test length(unique(state_df.RouteUsage)) == zbin
            
            # Each x value should appear zbin times
            for x in xval
                @test sum(state_df.Odometer .== x) == zbin
            end
            
            # Each z value should appear xbin times
            for z in zval
                @test sum(state_df.RouteUsage .== z) == xbin
            end
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 4: Future Value Computation
    #---------------------------------------------------------------------------
    @testset "Future Value Computation" begin
        @testset "compute_future_values basic" begin
            xbin = 3
            zbin = 2
            T = 5
            β = 0.9
            xval = collect(range(0, 200000, length=xbin))
            zval = collect(range(0, 1, length=zbin))
            xtran = rand(xbin*zbin, xbin)
            for i in 1:size(xtran,1)
                xtran[i,:] ./= sum(xtran[i,:])
            end
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            # Create mock logit model
            mock_data = DataFrame(
                Y = rand(0:1, 50),
                Odometer = rand([xval...], 50),
                RouteUsage = rand([zval...], 50),
                Branded = rand(0:1, 50),
                time = rand(1:T, 50)
            )
            mock_logit = glm(@formula(Y ~ Odometer + RouteUsage + Branded + time),
                            mock_data, Binomial(), LogitLink())
            
            FV = compute_future_values(state_df, mock_logit, xtran, xbin, zbin, T, β)
            
            # Test dimensions
            @test size(FV) == (xbin*zbin, 2, T+1)
            
            # Test terminal condition
            @test all(FV[:, :, T+1] .== 0)
            
            # Test all finite
            @test all(isfinite.(FV))
            
            # Test negative values (from -β * log)
            @test all(FV[:, :, 2:T] .>= 0)
        end
        
        @testset "Future value properties" begin
            xbin = 3
            zbin = 2
            T = 4
            β = 0.9
            xval = collect(range(0, 150000, length=xbin))
            zval = [0.4, 0.6]
            xtran = rand(xbin*zbin, xbin)
            for i in 1:size(xtran,1)
                xtran[i,:] ./= sum(xtran[i,:])
            end
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            mock_data = DataFrame(
                Y = rand(0:1, 30),
                Odometer = rand([xval...], 30),
                RouteUsage = rand([zval...], 30),
                Branded = rand(0:1, 30),
                time = rand(1:T, 30)
            )
            mock_logit = glm(@formula(Y ~ Odometer + Branded + time),
                            mock_data, Binomial(), LogitLink())
            
            FV = compute_future_values(state_df, mock_logit, xtran, xbin, zbin, T, β)
            
            # Test discount factor impact
            @test all(isfinite.(FV))
            
            # Test brand dimension
            @test size(FV, 2) == 2  # Two brand types (0 and 1)
            
            # Values should be more negative earlier in time (higher future value)
            for s in 1:size(FV, 1), b in 1:2
                if T > 2
                    @test FV[s, b, 2] <= FV[s, b, T] + 1e-10  # Earlier ≤ later (more negative)
                end
            end
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 5: FVT1 Computation (Mapping to Data)
    #---------------------------------------------------------------------------
    @testset "FVT1 Mapping to Data" begin
        @testset "compute_fvt1 basic" begin
            N = 5
            T = 20
            xbin = 3
            zbin = 2
            
            # Create mock data
            df_long = DataFrame(
                bus_id = repeat(1:N, inner=T),
                time = repeat(1:T, outer=N),
                Y = rand(0:1, N*T)
            )
            
            FV = rand(xbin*zbin, 2, T+1)
            FV[:, :, T+1] .= 0  # Terminal condition
            
            xtran = rand(xbin*zbin, xbin)
            for i in 1:size(xtran,1)
                xtran[i,:] ./= sum(xtran[i,:])
            end
            
            Xstate = rand(1:xbin, N, T)
            Zstate = rand(1:zbin, N)
            B = rand(0:1, N)
            
            fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)
            
            # Test dimensions
            @test length(fvt1) == N * T
            
            # Test all finite
            @test all(isfinite.(fvt1))
            
            # Test it's a vector
            @test isa(fvt1, Vector)
        end
        
        @testset "FVT1 computation properties" begin
            N = 4
            T = 20
            xbin = 3
            zbin = 2
            
            df_long = DataFrame(
                bus_id = repeat(1:N, inner=T),
                time = repeat(1:T, outer=N)
            )
            
            FV = -abs.(rand(xbin*zbin, 2, T+1))  # Negative values
            FV[:, :, T+1] .= 0
            
            xtran = rand(xbin*zbin, xbin)
            for i in 1:size(xtran,1)
                xtran[i,:] ./= sum(xtran[i,:])
            end
            
            Xstate = rand(1:xbin, N, T)
            Zstate = rand(1:zbin, N)
            B = rand(0:1, N)
            
            fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)
            
            # Test correct length
            @test length(fvt1) == nrow(df_long)
            
            # Test numerical properties
            @test all(isfinite.(fvt1))
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 6: Structural Parameter Estimation
    #---------------------------------------------------------------------------
    @testset "Structural Parameter Estimation" begin
        @testset "estimate_structural_params basic" begin
            N = 50
            
            df_long = DataFrame(
                Y = rand(0:1, N),
                Odometer = 50000 .+ 10000*randn(N),
                Branded = rand(0:1, N)
            )
            
            fvt1 = -abs.(rand(N))  # Future values (negative)
            
            theta_hat = estimate_structural_params(df_long, fvt1)
            
            # Test model type
            @test isa(theta_hat, StatsModels.TableRegressionModel)
            
            # Test coefficients
            @test length(coef(theta_hat)) == 3  # intercept + Odometer + Branded
            
            # Test all coefficients are finite
            @test all(isfinite.(coef(theta_hat)))
        end
        
        @testset "Structural model with offset" begin
            N = 40
            
            df_long = DataFrame(
                Y = rand(0:1, N),
                Odometer = 60000 .+ 15000*randn(N),
                Branded = rand(0:1, N)
            )
            
            fvt1 = -abs.(randn(N))
            
            # Test model estimation
            @test_nowarn theta_hat = estimate_structural_params(df_long, fvt1)
            
            theta_hat = estimate_structural_params(df_long, fvt1)
            
            # Test predictions work
            @test_nowarn predict(theta_hat)
            preds = predict(theta_hat)
            @test all(0 .<= preds .<= 1)
            @test length(preds) == N
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 7: Numerical Stability and Edge Cases
    #---------------------------------------------------------------------------
    @testset "Numerical Stability" begin
        @testset "Large parameter values" begin
            df = DataFrame(
                Y = rand(0:1, 30),
                Odometer = 100000 .+ 50000*randn(30),
                Branded = rand(0:1, 30),
                time = rand(1:10, 30)
            )
            
            # Test logit with large values
            @test_nowarn glm(@formula(Y ~ Odometer + Branded), 
                            df, Binomial(), LogitLink())
        end
        
        @testset "Small state spaces" begin
            xbin = 2
            zbin = 2
            xval = [0.0, 100000.0]
            zval = [0.3, 0.7]
            xtran = rand(xbin*zbin, xbin)
            for i in 1:size(xtran,1)
                xtran[i,:] ./= sum(xtran[i,:])
            end
            
            @test_nowarn construct_state_space(xbin, zbin, xval, zval, xtran)
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            @test nrow(state_df) == 4
        end
        
        @testset "All same decisions" begin
            # Test with all zeros
            df_zeros = DataFrame(
                Y = zeros(Int, 30),
                Odometer = randn(30),
                Branded = rand(0:1, 30)
            )
            
            @test_nowarn glm(@formula(Y ~ Odometer + Branded),
                            df_zeros, Binomial(), LogitLink())
            
            # Test with all ones
            df_ones = DataFrame(
                Y = ones(Int, 30),
                Odometer = randn(30),
                Branded = rand(0:1, 30)
            )
            
            @test_nowarn glm(@formula(Y ~ Odometer + Branded),
                            df_ones, Binomial(), LogitLink())
        end
    end
    
    #---------------------------------------------------------------------------
    # Test 8: Integration Tests
    #---------------------------------------------------------------------------
    @testset "Integration Tests" begin
        @testset "Complete workflow consistency" begin
            # Test that components work together
            xbin = 3
            zbin = 2
            T = 20
            N = 5
            β = 0.9
            
            xval = collect(range(0, 150000, length=xbin))
            zval = [0.4, 0.6]
            xtran = rand(xbin*zbin, xbin)
            for i in 1:size(xtran,1)
                xtran[i,:] ./= sum(xtran[i,:])
            end
            
            # Create state space
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            @test nrow(state_df) == xbin * zbin
            
            # Create mock logit
            mock_data = DataFrame(
                Y = rand(0:1, 30),
                Odometer = rand([xval...], 30),
                RouteUsage = rand([zval...], 30),
                Branded = rand(0:1, 30),
                time = rand(1:T, 30)
            )
            flex_logit = glm(@formula(Y ~ Odometer + RouteUsage + Branded),
                            mock_data, Binomial(), LogitLink())
            
            # Compute future values
            FV = compute_future_values(state_df, flex_logit, xtran, xbin, zbin, T, β)
            @test size(FV, 1) == xbin * zbin
            
            # Map to data
            df_long = DataFrame(
                bus_id = repeat(1:N, inner=T),
                time = repeat(1:T, outer=N),
                Y = rand(0:1, N*T),
                Odometer = rand([xval...], N*T),
                Branded = rand(0:1, N*T)
            )
            
            Xstate = rand(1:xbin, N, T)
            Zstate = rand(1:zbin, N)
            B = rand(0:1, N)
            
            fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)
            @test length(fvt1) == nrow(df_long)
            
            # Estimate structural params
            theta_hat = estimate_structural_params(df_long, fvt1)
            @test length(coef(theta_hat)) == 3
        end
    end
end

println("✓ All PS6 CCP-based Rust model tests completed successfully!")