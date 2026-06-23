#=-----------------------------------------------------------------------------#
    combine into video with:
        ffmpeg -r 30 -i check%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
#-----------------------------------------------------------------------------=#
using Images
using SparseArrays
using Plots

function clip_arr!(arr, threshold)

    for i = 1:length(arr)
        if arr[i] > threshold
            arr[i] = 1
        else
            arr[i] = 0
        end
    end
end

function make_scene(num_qubit,num_ancilla,M,t)

    # num_qubit = 7
    # num_ancilla = 3
    # M = 2
    
    d = 2^num_qubit
    d_ancilla = 2^num_ancilla

    # matFtoX, matFtoP, array_x = map_from_fock_to_position_or_momentum(d)
    matI = Matrix{Float64}(I, d, d)
    matI = sparse(matI)

    # define initail state rho0
    rho0 = zeros(ComplexF64, d, d)
    for n = 1:2^M
        rho0[n,n] = 1.0
    end
    rho0 .= rho0/tr(rho0)
    rho0 = sparse(rho0)

    # duplicate
    ancilla = [1 0; 0 1]/2
    ancilla = sparse(ancilla)
    for n = 1:num_ancilla
        rho0 = kron(ancilla,rho0)
    end
    rhot = copy(rho0)

    # expand nucleus?

    # squish electrons
    # t0 = time()
    # println("squish electrons")
    matSx3 = copy(matI)
    for n = 2:d_ancilla
        # println("n=",n)
        if n != 8
            # matSx3 = BlockDiagonal([matSx3,shearing_x2(d,1/3)])
            matSx3 = blockdiag(matSx3,sparse(shearing_x2(d,1/3)))
        else
            # matSx3 = BlockDiagonal([matSx3,matI])
            matSx3 = blockdiag(matSx3,matI)
        end
    end

    # matSx3 = BlockDiagonal([matI,shearing_x2(d,1/3),shearing_x2(d,1/3),shearing_x2(d,1/3)])
    rhot .= matSx3*rhot*matSx3'
    # println("total time:",round(time()-t0; digits = 3),"sec")
    
    # displace electrons
    # t0 = time()
    # println("displace electrons")
    matD2 = copy(matI)
    for n = 2:d_ancilla
        # println("n=",n)
        if n != 8
            # matD2 = BlockDiagonal([matD2,displacement(d,-1im*8)])
            matD2 = blockdiag(matD2,sparse(displacement(d,-1im*8)))
        else
            # matD2 = BlockDiagonal([matD2,matI])
            matD2 = blockdiag(matD2,matI)
        end
    end
    # matD2 = BlockDiagonal([matI,displacement(d,-1im*8),displacement(d,-1im*8),displacement(d,-1im*8)])
    rhot .= matD2*rhot*matD2'
    # println("total time:",round(time()-t0; digits = 3),"sec")

    # rotate electrons
    # t0 = time()
    # println("rotate electrons")
    matR2 = copy(matI)
    for n = 2:d_ancilla
        # println("n=",n)
        if n != 8
            # matR2 = BlockDiagonal([matR2,rotation(d,pi*t+pi/3*(n-2))])
            matR2 = blockdiag(matR2,sparse(rotation(d,pi*t+pi/3*(n-2))))
        else
            # matR2 = BlockDiagonal([matR2,matI])
            matR2 = blockdiag(matR2,matI)
        end
    end
    # matR2 = BlockDiagonal([matI,rotation(d,pi*time),rotation(d,pi*time+pi/3),rotation(d,pi*time+pi/3*2)])
    rhot .= matR2*rhot*matR2'
    # println("total time:",round(time()-t0; digits = 3),"sec")

    # stretch electrons
    # t0 = time()
    # println("stretch electrons")
    matSq2 = copy(matI)
    for n = 2:d_ancilla
        # println("n=",n)
        if n != 8
            # matSq2 = BlockDiagonal([matSq2,squeezing(d,log(1.5))])
            matSq2 = blockdiag(matSq2,sparse(squeezing(d,log(1.5))))
        else
            # matSq2 = BlockDiagonal([matSq2,matI])
            matSq2 = blockdiag(matSq2,matI)
        end
    end
    # matSq2 = BlockDiagonal([matI,squeezing(d,log(1.5)),squeezing(d,log(1.5)),squeezing(d,log(1.5))])
    rhot .= matSq2*rhot*matSq2'
    # println("total time:",round(time()-t0; digits = 3),"sec")

    # rotate electrons
    # t0 = time()
    # println("rotate electrons")
    matR2 = copy(matI)
    for n = 2:d_ancilla
        # println("n=",n)
        if n in [2,3,4]
            # matR2 = BlockDiagonal([matR2,rotation(d,pi/3*(n-2))])
            matR2 = blockdiag(matR2,sparse(rotation(d,pi/3*(n-2))))
        elseif n in [5,6,7]
            matR2 = blockdiag(matR2,sparse(rotation(d,pi/3*(n-5))))
        elseif n == 8
            # matR2 = BlockDiagonal([matR2,matI])
            matR2 = blockdiag(matR2,matI)
        end
    end
    # matR2 = BlockDiagonal([matI,matI,rotation(d,pi/3),rotation(d,pi/3*2)])
    rhot .= matR2*rhot*matR2'
    # println("total time:",round(time()-t0; digits = 3),"sec")

    # trace out ancillas
    rhot_ptraceout = zeros(ComplexF64, d, d)
    for i = 1:d
        for j = 1:d
            for n = 1:d_ancilla
                rhot_ptraceout[i,j] += rhot[i+d*(n-1),j+d*(n-1)]
                # rhot_ptraceout[i,j] = rhot[i,j] + rhot[i+d,j+d] + rhot[i+2*d,j+2*d] + + rhot[i+3*d,j+3*d]
            end
        end
    end
    rhot = rhot_ptraceout

    return rhot

    # plot husimi
    println(tr(rhot))
    
    q_range = copy(array_x) 
    # q_range = collect(range(0.0, 10.0, length=N))
    p_range = copy(q_range)
    mat_husimi = prepation_husimi_fock(q_range, p_range, d)
    f_husimi = husimi(d,rhot,mat_husimi,q_range,p_range)

    figure()
    pcolor(q_range, p_range, abs.(f_husimi)')
    colorbar()

    # figure()
    # plot(abs.(f_husimi[:,Int64(d/2)]))

end

function test_make_scene_on_husimi(t)

    num_qubit = 7
    num_ancilla = 3
    M = 2
    d = 2^num_qubit
    # t = 0.2

    matFtoX, matFtoP, array_x = map_from_fock_to_position_or_momentum(d)
    q_range = copy(array_x)
    p_range = copy(q_range)
    mat_husimi = prepation_husimi_fock(q_range, p_range, d)

    rhot = make_scene(num_qubit,num_ancilla,M,t)
    f_husimi = husimi(d,rhot,mat_husimi,q_range,p_range)

    # figure()
    # pcolor(q_range, p_range, abs.(f_husimi)')
    # colorbar()

    heatmap(abs.(f_husimi)'; aspect_ratio=1, colorbar=false, axis=false)

end

function make_animation(; final_time = 1, clip = false, threshold = 0.5)

    # system parameters
    num_qubit = 8
    num_ancilla = 3
    M = 2
    d = 2^num_qubit

    # for husimi
    matFtoX, matFtoP, array_x = map_from_fock_to_position_or_momentum(d)
    q_range = copy(array_x)
    p_range = copy(q_range)
    mat_husimi = prepation_husimi_fock(q_range, p_range, d)
    rhot = zeros(ComplexF64, d, d)
    f_husimi = zeros(ComplexF64, d, d)
    # output_image = zeros(1000,1000)

    # for a video
    fps = 30
    time = 0

    i = 0
    while time < final_time - 1/fps
    #while time < 1/fps


        # time evolution
        rhot .= make_scene(num_qubit,num_ancilla,M,time)
        f_husimi .= husimi(d,rhot,mat_husimi,q_range,p_range)

        output_arr = Array(abs.(f_husimi)')
        if clip
            clip_arr!(output_arr, threshold)
        end

        output_image = heatmap(output_arr;
                               aspect_ratio=1, colorbar=false, axis=false,
                               size=(1024, 1024))
        filename = "check"*lpad(i, 4, "0")*".png"
        println(filename)
        save(filename, output_image)

        i += 1
        time += 1/fps

    end 
end
