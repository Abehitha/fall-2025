using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

cd(@__DIR__)

Random.seed!(1234)

include("PS4-Abehitha_source.jl")

allwrap()