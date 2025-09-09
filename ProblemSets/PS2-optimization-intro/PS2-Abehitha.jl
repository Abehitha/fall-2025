#=  Problem Set 2
    Author: Abehitha
    Date: 2025-09-09
    Description: This script contains the solutions to Problem Set 2
=#

using Optim, HTTP, GLM, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables

cd(@__DIR__)

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