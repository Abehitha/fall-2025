#=  Unit Tests for Problem Set 1: Working with Julia Code
    Author: Test Suite
    Date: 2025-09-04
    Description: Comprehensive unit tests for PS1-Abehitha.jl functions
=#

using Test, JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

# Include the main file
include("PS1-Abehitha.jl")

@testset "Problem Set 1 Tests" begin
    
    @testset "Question 1 (q1) Tests" begin
        # Set seed for reproducible tests
        Random.seed!(1234)
        
        # Test q1 function
        A, B, C, D = q1()
        
        @testset "Matrix A Tests" begin
            @test size(A) == (10, 7)
            @test eltype(A) == Float64
            @test all(A .>= -5) && all(A .<= 10)  # Check bounds for uniform distribution
        end
        
        @testset "Matrix B Tests" begin
            @test size(B) == (10, 7)
            @test eltype(B) == Float64
            # For normal distribution, we can't guarantee exact bounds, but check reasonable range
            @test length(B) == 70
        end
        
        @testset "Matrix C Tests" begin
            @test size(C) == (5, 7)  # Should be [A[1:5, 1:5] B[1:5, 6:7]]
            @test eltype(C) == Float64
        end
        
        @testset "Matrix D Tests" begin
            @test size(D) == (10, 7)
            @test eltype(D) == Float64
            # D should contain only non-positive values from A or zeros
            @test all(D .<= 0)
        end
        
        @testset "File Creation Tests" begin
            @test isfile("matrixpractice.jld")
            @test isfile("firstmatrix.jld")
            @test isfile("Cmatrix.csv")
            @test isfile("Dmatrix.dat")
            
            # Test that saved matrices can be loaded
            saved_data = load("firstmatrix.jld")
            @test haskey(saved_data, "A")
            @test haskey(saved_data, "B")
            @test haskey(saved_data, "C")
            @test haskey(saved_data, "D")
        end
    end
    
    @testset "Question 2 (q2) Tests" begin
        # Create test matrices
        Random.seed!(1234)
        test_A = rand(5, 3)
        test_B = rand(5, 3)
        test_C = -10 .+ 20 * rand(4, 4)  # Range from -10 to 10
        
        @testset "q2 Execution Test" begin
            # Test that q2 runs without error
            @test_nowarn q2(test_A, test_B, test_C)
        end
        
        @testset "Element-wise Operations Test" begin
            # Test the element-wise multiplication logic manually
            AB_manual = zeros(size(test_A))
            for r in 1:size(test_A, 1)
                for c in 1:size(test_A, 2)
                    AB_manual[r, c] = test_A[r, c] * test_B[r, c]
                end
            end
            AB_vectorized = test_A .* test_B
            @test AB_manual ≈ AB_vectorized
        end
        
        @testset "Filtering Operations Test" begin
            # Test the filtering logic manually
            test_small_C = [-6.0 -3.0; 0.0 8.0; 2.0 -1.0]
            filtered_manual = Float64[]
            for c in 1:size(test_small_C, 2)
                for r in 1:size(test_small_C, 1)
                    if test_small_C[r, c] >= -5 && test_small_C[r, c] <= 5
                        push!(filtered_manual, test_small_C[r, c])
                    end
                end
            end
            filtered_vectorized = test_small_C[(test_small_C .>= -5) .& (test_small_C .<= 5)]
            @test Set(filtered_manual) == Set(filtered_vectorized)  # Using Set to ignore order
        end
    end
    
    @testset "Question 4 (matrixops) Tests" begin
        @testset "Compatible Matrices" begin
            test_A = [1.0 2.0; 3.0 4.0]
            test_B = [2.0 1.0; 1.0 2.0]
            
            out1, out2, out3 = matrixops(test_A, test_B)
            
            # Test element-wise product
            @test out1 ≈ [2.0 2.0; 3.0 8.0]
            
            # Test matrix product A' * B
            expected_out2 = [1.0 2.0; 3.0 4.0]' * [2.0 1.0; 1.0 2.0]
            @test out2 ≈ expected_out2
            
            # Test sum of all elements of A + B
            @test out3 ≈ sum([3.0 3.0; 4.0 6.0])
        end
        
        @testset "Incompatible Matrices" begin
            test_A = [1.0 2.0; 3.0 4.0]
            test_B = [1.0 2.0 3.0; 4.0 5.0 6.0]
            
            @test_throws ErrorException matrixops(test_A, test_B)
        end
        
        @testset "Type Requirements" begin
            # Test with wrong types
            test_A_int = [1 2; 3 4]
            test_B_float = [1.0 2.0; 3.0 4.0]
            
            @test_throws MethodError matrixops(test_A_int, test_B_float)
        end
    end
    
    @testset "Data Structure Tests" begin
        @testset "Array Operations" begin
            # Test basic array operations used in the code
            test_matrix = rand(3, 4)
            
            # Test reshaping operations
            @test length(vec(test_matrix)) == length(test_matrix)
            @test size(reshape(test_matrix, 12, 1)) == (12, 1)
            @test size(reshape(test_matrix, length(test_matrix), 1)) == (12, 1)
            
            # Test concatenation
            test_matrix2 = rand(3, 4)
            cat_result = cat(test_matrix, test_matrix2; dims=3)
            @test size(cat_result) == (3, 4, 2)
            
            # Test permutation
            perm_result = permutedims(cat_result, (3, 1, 2))
            @test size(perm_result) == (2, 3, 4)
        end
        
        @testset "Kronecker Product" begin
            A = [1.0 2.0; 3.0 4.0]
            B = [0.0 1.0; 1.0 0.0]
            K = kron(A, B)
            
            @test size(K) == (4, 4)
            # Test some properties of Kronecker product
            @test K[1:2, 1:2] ≈ A[1,1] * B
            @test K[1:2, 3:4] ≈ A[1,2] * B
        end
    end
    
    @testset "Random Number Generation Tests" begin
        Random.seed!(1234)
        
        @testset "Uniform Distribution" begin
            uniform_vals = rand(Uniform(-5, 10), 100)
            @test all(uniform_vals .>= -5)
            @test all(uniform_vals .<= 10)
        end
        
        @testset "Normal Distribution" begin
            normal_vals = rand(Normal(-2, 15), 100)
            @test length(normal_vals) == 100
            # Test that mean is approximately correct (with some tolerance)
            @test abs(mean(normal_vals) - (-2)) < 5  # Allow some variance
        end
        
        @testset "Binomial Distribution" begin
            binomial_vals = [rand(Binomial(20, 0.6)) for _ in 1:100]
            @test all(binomial_vals .>= 0)
            @test all(binomial_vals .<= 20)
        end
    end
    
    @testset "File I/O Tests" begin
        @testset "CSV Operations" begin
            # Create a test matrix
            test_matrix = [1.0 2.0 3.0; 4.0 5.0 6.0]
            test_df = DataFrame(test_matrix, :auto)
            
            # Test CSV writing and reading
            test_filename = "test_matrix.csv"
            CSV.write(test_filename, test_df)
            @test isfile(test_filename)
            
            # Read back and compare
            read_df = DataFrame(CSV.File(test_filename))
            @test size(read_df) == size(test_df)
            
            # Cleanup
            rm(test_filename)
        end
        
        @testset "JLD Operations" begin
            # Test saving and loading with JLD
            test_array = [1.0 2.0; 3.0 4.0]
            test_filename = "test_data.jld"
            
            save(test_filename, "test_matrix", test_array)
            @test isfile(test_filename)
            
            loaded_data = load(test_filename)
            @test haskey(loaded_data, "test_matrix")
            @test loaded_data["test_matrix"] ≈ test_array
            
            # Cleanup
            rm(test_filename)
        end
    end
end

# Helper function to run all tests
function run_all_tests()
    println("Running all tests for PS1...")
    Test.run_tests()
    println("All tests completed!")
end

# Cleanup function to remove test files
function cleanup_test_files()
    test_files = ["matrixpractice.jld", "firstmatrix.jld", "Cmatrix.csv", "Dmatrix.dat", "nlsw88_cleaned.csv"]
    for file in test_files
        if isfile(file)
            rm(file)
            println("Removed $file")
        end
    end
end

println("Test suite loaded. Run `run_all_tests()` to execute all tests.")
println("Run `cleanup_test_files()` to remove generated files after testing.")
