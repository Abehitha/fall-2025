#=  Problem Set 1: Working with Julia Code
    Author: Abehitha
    Date: 2025-08-31
    Description: This script contains the solutions to Problem Set 1 
=#

using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

function q1()
    #===================================================================================
    *************************     Part 1   *********************************************
    ===================================================================================#

    #=========================== Part 1: a ============================================#

    # Set the seed
    Random.seed!(1234)

    #*****Part 1: a(i)*******#

    # Draw uniform random numbers
    A = -5 .+ 15 * rand(10, 7)
    A = rand(Uniform(-5, 10), 10, 7)

    #*****Part 1: a(ii)*******#

    # Draw normal random numbers
    B = -2 .+ 15 * randn(10, 7)
    B = rand(Normal(-2, 15), 10, 7)

    #*****Part 1: a(iii)*******#

    # Indexing
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    C = [A[1:5, 1:5] B[1:5, end-1:end]]

    #*****Part 1: a(iv)*******#

    #Bit Array/ Dummy variable
    D = A .* (A .<= 0)

    #=========================== Part 1: b ============================================#

    size(A)
    size(A, 1) * size(A, 2)
    length(A)
    size(A[:])

    #=========================== Part 1: c ============================================#

    length(D)
    length(unique(D))

    #=========================== Part 1: d ============================================#
    E = reshape(B, 70, 1)
    E = reshape(B, (70, 1))
    E = reshape(B, length(B), 1)
    E = reshape(B, size(B, 1) * size(B, 2), 1)
    E = B[:]
    E = vec(B)

    #=========================== Part 1: e ============================================#

    F = cat(A, B; dims=3)

    #=========================== Part 1: f ============================================#

    F = permutedims(F, (3, 1, 2))

    #=========================== Part 1: g ============================================#

    G = kron(B, C)
    #G = kron(C, F) This doesn't work. Kronecker product is defined only for 2D matrices

    #=========================== Part 1: h ============================================#

    #save matrices A, B, C, D, E, F and G as a .jld file named matrixpractice
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)
    @save "matrixpractice.jld" A B C D E F G

    #=========================== Part 1: i ============================================#

    #save only the matrices A, B, C, and D as a .jld file called firstmatrix.
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)
    @save "firstmatrix.jld" A B C D

    #=========================== Part 1: j ============================================#

    #Export C as a .csv file called Cmatrix.
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))

    #=========================== Part 1: k ============================================#

    dfD = DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", dfD, delim='\t')
    #Through piping: Allows not to create a new variable
    DataFrame(D, :auto) |> CSV.write("Dmatrix.dat", delim='\t')

    #=========================== Part 1: l ============================================#

    return A, B, C, D
end

#call the function from q1
A, B, C, D = q1()