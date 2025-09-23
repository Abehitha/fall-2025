
using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS3-Abehitha_source.jl")

allwrap()

#Question: Interpret the coefficient gamma
# Estimated gamma is -0.094
# Gamma represents the change in latent utility with a 1-unit change in the relative E(log wage) in occupation j (relative to Other).
# It's surprising that gamma is negative, as we would expect higher wages to increase utility and choice probability.