#=------------classical.jl-----------------------------------------------------#
 Purpose: This file is meant to cover all the classical methods from the 
          QIFS paper: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.68.046110

   Notes: Examples 1-3
#-----------------------------------------------------------------------------=#

cantor_1(x) = x/3

cantor_2(x) = x/3 + 2/3

function example_1()
    pt = rand()
    res = 1000
    arr = zeros(res)
    for i = 1:1000000
        pt = rand((cantor_1, cantor_2))(pt)
        bin = floor(Int, pt*res)+1
        arr[bin] += 1
    end

    return arr
end

function select_fx_num(probs::Tuple)
    if !isapprox(sum(probs), 1)
        error("probs don't add up!")
    end

    rng = rand()
    level = 0
    for i = 1:length(probs)-1
        level += probs[i]
        if rng < level
            return i
        end
    end

    return length(probs)
end

function example_2()
    pt = rand()
    res = 1000
    arr = zeros(res)
    choice = 0
    fxs = (cantor_1, cantor_2)
    for i = 1:10000000
        choice = select_fx_num((pt, 1-pt))
        pt = fxs[choice](pt)
        bin = floor(Int, pt*res)+1
        arr[bin] += 1
    end

    return arr
end
