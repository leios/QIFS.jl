#=------------classical.jl-----------------------------------------------------#
 Purpose: This file is meant to cover all the classical methods from the 
          QIFS paper: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.68.046110

   Notes: Examples 1-5;
          4 is nontrivial as it requires projections on to 2D
          5 is does not converge
#-----------------------------------------------------------------------------=#

cantor_1(x) = x/3

cantor_2(x) = x/3 + 2/3

#=-----------------------------------------------------------------------------#
EXAMPLE 1
#-----------------------------------------------------------------------------=#

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

#=-----------------------------------------------------------------------------#
EXAMPLE 2
#-----------------------------------------------------------------------------=#

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

#=-----------------------------------------------------------------------------#
EXAMPLE 3
#-----------------------------------------------------------------------------=#

struct Point{FT <: Real}
    x::FT
    y::FT
end

function tartan_1(pt::Point)
    out = [1/3 0.; 0. 1.]*[pt.x, pt.y]
    return Point(out[1], out[2])
end

function tartan_2(pt::Point)
    out = [1/3 0.; 0. 1.]*[pt.x, pt.y] .+ [2/3, 0]
    return Point(out[1], out[2])
end

function tartan_3(pt::Point)
    out = [1. 0.; 0. 1/3]*[pt.x, pt.y]
    return Point(out[1], out[2])
end

function tartan_4(pt::Point)
    out = [1. 0.; 0. 1/3]*[pt.x, pt.y] .+ [0, 2/3]
    return Point(out[1], out[2])
end

function example_3()
    pt = Point(rand(), rand())
    res = 1000
    arr = zeros(res, res)
    choice = 0
    fxs = (tartan_1, tartan_2, tartan_3, tartan_4)
    for i = 1:10000000
        pt = rand(fxs)(pt)
        bin_x = floor(Int, pt.x*res)+1
        bin_y = floor(Int, pt.y*res)+1
        arr[bin_x, bin_y] += 1
    end

    return arr
end

#=-----------------------------------------------------------------------------#
EXAMPLE_5
#-----------------------------------------------------------------------------=#

tent_map(x) = x < 0.5 ? 2*x : 2*(1-x)
bernoulli_map(x) = x < 0.5 ? 2*x : 2x-1

function example_5()
    pt = rand()
    res = 100
    arr = zeros(res)
    for i = 1:1000
        pt = rand((tent_map, bernoulli_map))(pt)
        bin = floor(Int, pt*res)+1
        arr[bin] += 1
    end

    return arr
end

#=-----------------------------------------------------------------------------#
AUX
#-----------------------------------------------------------------------------=#

function halfway(P, A)
    return Point((P.x + A.x)*0.5, (P.y + A.y)*0.5)
end

function generate_square(A, B, C, D)

    pt = Point(rand(), rand())
    res = 10
    arr = zeros(res, res)
    for i = 1:1000
        choice = rand((A, B, C, D))
        pt = halfway(pt, choice)
        bin_x = floor(Int, pt.x*res)+1
        bin_y = floor(Int, pt.y*res)+1
        arr[bin_x, bin_y] += 1
    end 

    return arr
end

function baker_1(P; a = 0.5)
    #return Point(2 * P.x, a*P.y)
    #return Point(0.5 * P.x, a*P.y)
    return Point(0.5*(P.x + floor(2*P.y)), 2*P.y - floor(2*P.y))
end

function baker_2(P; a = 0.5)
    #return Point(2 * P.x - 1, a*P.y + 0.5)
    return Point(0.5 * P.x + 0.5, a*P.y + 0.5)
end

#=
mkdir res/

julia> for i = 1:100
           arr = QIFS.generate_square_baker(i)
           filename = "res/check"*lpad(i, 4, "0")*".png"
           savefig(heatmap(arr), filename)
       end

ffmpeg -i res/check0%04d.png out.mp4
=#

function generate_square_baker(num_iterations)

    pt = Point(rand(), rand())
    pts = [Point(rand(), rand()) for i = 1:100]
    res = 10
    arr = zeros(res, res)
    for i = 1:num_iterations
#=
        if (pt.x < 0.5)
            fx = baker_1
        else
            fx = baker_2
        end
        fx = rand((baker_1, baker_2))
=#
        pts = baker_1.(pts)
        for i = 1:100
            bin_x = floor(Int, (pts[i].x*0.5 + 0.25)*res)+1
            bin_y = floor(Int, (pts[i].y*0.5 + 0.25)*res)+1
            arr[bin_x, bin_y] += 1
        end
    end

    return arr
end

