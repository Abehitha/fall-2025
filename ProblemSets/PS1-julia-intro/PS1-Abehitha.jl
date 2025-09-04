#=  Problem Set 1: Working with Julia Code
    Author: Abehitha
    Date: 2025-08-31
    Description: This script contains the solutions to Problem Set 1 
=#

using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

function q1()
    #===============================================================================
    *************************     Part 1   *****************************************
    ===============================================================================#

    #=========================== Part 1: a ========================================#

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

    #=========================== Part 1: b ========================================#

    size(A)
    size(A, 1) * size(A, 2)
    length(A)
    size(A[:])

    #=========================== Part 1: c ========================================#

    length(D)
    length(unique(D))

    #=========================== Part 1: d ========================================#
    E = reshape(B, 70, 1)
    E = reshape(B, (70, 1))
    E = reshape(B, length(B), 1)
    E = reshape(B, size(B, 1) * size(B, 2), 1)
    E = B[:]
    E = vec(B)

    #=========================== Part 1: e ========================================#

    F = cat(A, B; dims=3)

    #=========================== Part 1: f ========================================#

    F = permutedims(F, (3, 1, 2))

    #=========================== Part 1: g ========================================#

    G = kron(B, C)
    #G = kron(C, F) Doesn't work. Kronecker product is defined only for 2D matrices

    #=========================== Part 1: h ========================================#

    #save matrices A, B, C, D, E, F and G as a .jld file named matrixpractice
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)
    @save "matrixpractice.jld" A B C D E F G

    #=========================== Part 1: i ========================================#

    #save only the matrices A, B, C, and D as a .jld file called firstmatrix.
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)
    @save "firstmatrix.jld" A B C D

    #=========================== Part 1: j ========================================#

    #Export C as a .csv file called Cmatrix.
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))

    #=========================== Part 1: k ========================================#

    dfD = DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", dfD, delim='\t')
    #Through piping: Allows not to create a new variable
    DataFrame(D, :auto) |> CSV.write("Dmatrix.dat", delim='\t')

    #=========================== Part 1: l ========================================#

    return A, B, C, D
end


#===================================================================================
*************************     Part 2   *********************************************
===================================================================================#

#=========================== Part 2: a ============================================#

function q2(A, B, C)
    #=========================== Part 2: a ============================================#
    AB = zeros(size(A))
    for r in 1:size(A, 1)
        for c in 1:size(A, 2)
            AB[r, c] = A[r, c] * B[r, c]
        end
    end

    AB = A .* B

    #=========================== Part 2: b ============================================#
    # find indices of C where value of C is between -5 and 5
    Cprime = Float64[]
    for c in 1:size(C, 2)
        for r in 1:size(C, 1)
            if C[r, c] >= -5 && C[r, c] <= 5
                push!(Cprime, C[r, c])
            end
        end
    end

    Cprime2 = C[(C .>= -5) .& (C .<= 5)]

    # Compare the two vectors
    Cprime == Cprime2
    if Cprime != Cprime2
        @show size(Cprime)
        @show size(Cprime2)
        @show Cprime .== Cprime2
        error("Cprime and Cprime2 are not the same")
    end

    #=========================== Part 2: c ============================================#
    N = 15_169
    K = 6
    T = 5
    X = zeros(15_169,6,5)
    # ordering of the 2nd dimension:
    # intercept
    # dummy variable
    # continous variable (normal)
    # normal
    # binomial ("discrete" normal)
    # another binomial
    for i in axes(X,1)
        X[i,1,:] .= 1.0
        X[i,5,:] .= rand(Binomial(20,0.6))
        X[i,6,:] .= rand(Binomial(20,0.5))
        for t in axes(X,3)
            X[i,2,t] = rand() <= .75 * (6 - t)/5
            X[i,3,t] = rand(Normal(15+t-1,5*(t-1)))
            X[i,4,t] = rand(Normal(π*(6-t),1/exp(1)))
        end

    end

    #=========================== Part 2: d ============================================#
    # comprehension practice
    β = zeros(K, T)
    β[1,:] = [1+0.25*(t-1) for t in 1:T]
    β[2,:] = [log(t) for t in 1:T]
    β[3,:] = [-sqrt(t) for t in 1:T]
    β[4,:] = [exp(t) - exp(t+1) for t in 1:T]
    β[5,:] = [t for t in 1:T]
    β[6,:] = [t/3 for t in 1:T]

    #=========================== Part 2: e ============================================#
    Y = [X[:,:,t] * β[:,t] .+ rand(Normal(0,0.36), N) for t in 1:T]

    return nothing
end

function q3()
    #===================================================================================
    **********************     Part 3   *********************************************
    ===================================================================================#
    #=========================== Part 3: a ============================================#
    
    df = DataFrame(CSV.File("nlsw88.csv"))
    @show df[1:5, :]
    @show typeof(df[:, :grade])

    #=========================== Part 3: b ============================================#
    # percentage never married
    @show mean(df[:, :never_married])

    #=========================== Part 3: c ============================================#
    @show freqtable(df[:, :race])

    #=========================== Part 3: d ============================================#
    vars = names(df)
    summarystats = describe(df)
    @show summarystats

    #=========================== Part 3: e ============================================#
    # cross tabulation of industry and occupation
    @show freqtable(df[:, :industry], df[:, :occupation])

    #=========================== Part 3: f ============================================#
    #Tabulate the mean wage over industry and occupation categories. Hint: you should first
    #subset the data frame to only include the columns industry, occupation and wage. You
    #should then follow the “split-apply-combine” directions here.
    df_sub =df[:, [:industry, :occupation, :wage]]
    grouped = groupby(df_sub, [:industry, :occupation])
    mean_wage = combine(grouped, :wage => mean => :mean_wage)
    @show mean_wage

    return nothing
end

#call the function from q1
A, B, C, D = q1()

#call the function from q2
q2(A, B, C)
