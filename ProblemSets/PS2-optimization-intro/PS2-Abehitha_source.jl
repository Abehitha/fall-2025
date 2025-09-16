#=  Problem Set 2
    Author: Abehitha
    Date: 2025-09-09
    Description: This script contains the solutions to Problem Set 2
=#

using Optim, HTTP, GLM, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables

cd(@__DIR__)

function PS2()
#===============================================================================
*************************     Question 1   *************************************
===============================================================================#
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))

# this will result in better conversions since we are starting closer to the actual minimum
result_better = optimize(minusf, [-7.0], BFGS())
println(result_better)

#===============================================================================
*************************     Question 2   *************************************
===============================================================================#
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

#There are many ways to define the sum of squared residuals. 
function ols2(beta, X, y)
    ssr = sum((y.-X*beta).^2)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y

# standard errors
N=size(X,1)
K=size(X,2)
MSE = sum((y - X*bols).^2)/(N - K)
VCOV = MSE * inv(X'*X)
se_bols = sqrt.(diag(VCOV))
println("Standard errors: ", se_bols)


println("OLS closed form: ", bols)
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)

#===============================================================================
*************************     Question 3   *************************************
===============================================================================#
function logit(alpha, X, y)
    loglike=sum(y.*(X*alpha)-log.(1 .+exp.(X*alpha)))
    return -loglike
end

beta = optimize(alpha -> logit(alpha, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta.minimizer)

#===============================================================================
*************************     Question 4   *************************************
===============================================================================#

beta_glm=glm(@formula(married ~ age + white + collgrad),df, Binomial(),LogitLink())
println(beta_glm)

#===============================================================================
*************************     Question 5   *************************************
===============================================================================#
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

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

K = size(X, 2)
J = length(unique(y))
n_params = K * (J - 1)
alpha_zero = zeros(n_params)
alpha_rand1 = rand(n_params)
alpha_rand2 = 2 .* rand(n_params) .- 1
alpha_true = [.1910213, -.0335262, .5963968, .4165052, -.1698368, -.0359784, 1.30684, -.430997, .6894727, -.0104578, .5231634, -1.492475, -2.26748, -.0053001, 1.391402, -.9849661, -1.398468, -.0142969, -.0176531, -1.495123, .2454891, -.0067267, -.5382892, -3.78975]

println("Trying different starting values...")
println("\n1. Starting with zeros:")
result_zero = optimize(alpha -> mlogit(alpha, X, y), alpha_zero, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false))
println("Converged: ", Optim.converged(result_zero))
println("Final objective: ", result_zero.minimum)

println("\n2. Starting with random [0,1]:")
result_rand1 = optimize(alpha -> mlogit(alpha, X, y), 
                       alpha_rand1, 
                       LBFGS(), 
                       Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false))
println("Converged: ", Optim.converged(result_rand1))
println("Final objective: ", result_rand1.minimum)

println("\n3. Starting with random [-1,1]:")
result_rand2 = optimize(alpha -> mlogit(alpha, X, y), 
                       alpha_rand2, 
                       LBFGS(), 
                       Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false))
println("Converged: ", Optim.converged(result_rand2))
println("Final objective: ", result_rand2.minimum)

println("\n4. Starting with true values:")
result_true = optimize(alpha -> mlogit(alpha, X, y), 
                      alpha_true, 
                      LBFGS(), 
                      Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false))
println("Converged: ", Optim.converged(result_true))
println("Final objective: ", result_true.minimum)

results = [result_zero, result_rand1, result_rand2, result_true]
objectives = [r.minimum for r in results]
best_idx = argmin(objectives)
best_result = results[best_idx]

println("\n" * "="^60)
println("BEST RESULT: Starting values option $best_idx")
println("Final log-likelihood: ", round(best_result.minimum, digits=6))
println("Converged: ", Optim.converged(best_result))

beta_matrix = reshape(best_result.minimizer, K, J-1)
println("\nCoefficient Matrix (4 x 6):")
println("Rows: [Intercept, Age, Race==1, CollGrad==1]") 
println("Columns: Categories 1-6 vs Category 7 (reference)")
println()

row_names = ["Intercept", "Age", "Race==1", "CollGrad==1"]
col_names = ["Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5", "Cat 6"]

# Print header
print(rpad("", 12))
for j in 1:6
    print(lpad(col_names[j], 12))
end
println()

# Print coefficients
for i in 1:4
    print(rpad(row_names[i], 12))
    for j in 1:6
        print(lpad(round(beta_matrix[i,j], digits=6), 12))
    end
    println()
end

println("\nNote: Category 7 is the reference category with coefficients = 0")

return nothing
end

#call the function from q5
PS2()