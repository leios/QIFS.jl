using LinearAlgebra
using QuantumOptics
using PyPlot
using JLD2

#=-----------------------------------------------------------------------------#
EXAMPLE 8
#-----------------------------------------------------------------------------=#

function example_8()

    # define rho1 and rho2
    rho1 = zeros(Int64,2,2)
    rho1[1,1] = 1
    rho2 = zeros(Int64,2,2)
    rho2[2,2] = 1

    # define initail random pure state rho0
    psi0 = rand(ComplexF64,2)
    psi0 .= psi0/sqrt(sum(abs.(psi0).^2))
    rho0 = psi0 * psi0'

    rhok = rho0
    choice = rand()

    res = 1000
    arr = zeros(res)

    for k = 1:1000000

        if choice < 1/2
            rhok .= rhok/3 + rho1*2/3
        else
            rhok .= rhok/3 + rho2*2/3
        end

        bin = floor(Int, rhok[1,1]*res)+1
        arr[bin] += 1

        choice = rand()

    end

    return arr

end

#=-----------------------------------------------------------------------------#
EXAMPLE 14
#-----------------------------------------------------------------------------=#

function coherent_state(N::Int64,q::Float64,p::Float64)

    overlap_qp = zeros(ComplexF64,N)

    for n = 1:N
        
        for k = 1:N
            braket_kkappa = exp(-pi/N*(k-N/2)^2) * exp(-1im*k*pi)
            for j = 1:N
                overlap_qp[n] += braket_kkappa * exp(1im*2*pi*(-j*n + (j-N*p+N/2)*(k+N*q-N/2))/N)
            end
        end

    end
    overlap_qp .= overlap_qp/N/(N/2)^(1/4)

    return overlap_qp

end

function map_G1234!(rhot::Matrix{ComplexF64},rhot_p::Matrix{ComplexF64},rhot1::Matrix{ComplexF64},rhot1_p::Matrix{ComplexF64},mat::Matrix{ComplexF64},L::Int64,N::Int64)

    rhot_p .= mat*rhot*mat' # rhot in p basis
    rhot1 .= 0.0
    rhot1_p .= 0.0

    # see Eqs.(24-27) in PRE

    for i = 1:L
        for j = 1:L

            i1 = i
            j1 = j
            i2 = 2*L+i
            j2 = 2*L+j

            for n = 0:2
                rhot1[i1,j1] += rhot[mod1(3*i1+n,N),mod1(3*j1+n,N)]
                rhot1[i2,j2] += rhot[mod1(3*i2+n,N),mod1(3*j2+n,N)]
                rhot1_p[i1,j1] += rhot_p[mod1(3*i1+n,N),mod1(3*j1+n,N)]
                rhot1_p[i2,j2] += rhot_p[mod1(3*i2+n,N),mod1(3*j2+n,N)]
            end
            
        end
    end

    rhot .= rhot1*0.25 + mat'*rhot1_p*mat*0.25

end

function husimi_format(N::Int64)

    # generate <n|q,p> for all n,q,p for husimi

    vec_q = [0:1/N:1;]
    vec_p = copy(vec_q)
    overlap_qp_mat = zeros(ComplexF64,N,N+1,N+1)
    t0 = time()

    for jq = 1:N+1 # parallelisable
        for jp = 1:N+1
            # f_husimi[jq,jp] = husimi_overlap_0(N,rho,vec_q[jq],vec_p[jp])
            overlap_qp_mat[:,jq,jp] = husimi_overlap(N,vec_q[jq],vec_p[jp])
        end

        # show progress
        if mod(jq,10) == 0
            println(round(jq/(N+1)*100; digits = 3),"%")
            println(round(time()-t0; digits = 3),"sec")
        end

    end

    return overlap_qp_mat, vec_q, vec_p

end

function save_overlap_qp_mat()

    # "husimi_format" takes time, and so compute it once and save it

    L = 3^3
    N = 3*L
    overlap_qp_mat, vec_q, vec_p = husimi_format(N)

    # save
    jldsave("overlap_qp_mat_L27.jld2"; overlap_qp_mat)

end

function husimi(N::Int64,rho::Matrix{ComplexF64},overlap_qp_mat::Array{ComplexF64, 3})

    # see Eq.(28) in PRE

    vec_q = [0:1/N:1;]
    vec_p = copy(vec_q)
    f_husimi = zeros(ComplexF64,N+1,N+1)

    for jq = 1:N+1
        for jp = 1:N+1
            f_husimi[jq,jp] = (overlap_qp_mat[:,jq,jp])'*rho*overlap_qp_mat[:,jq,jp] # maybe /2/pi ?
        end
    end

    return f_husimi

end

function example_14()

    # see above Eq.(24)
    
    L = 3^3 #30 #3^2
    N = 3*L # system size
    Nt = 100 # number of iteration

    # define initail state rho0
    psi0 = coherent_state(N,0.25,0.25) # can be any state in principle but we choose a cherent state
    rho0 = psi0*psi0'

    # define density matrices
    rhot = copy(rho0) # time-evolved state to track
    rhot_p = copy(rho0)
    rhot1 = copy(rhot)
    rhot1_p = copy(rhot)
    
    # define matrix to transform q basis to p basis (complex conjugate transposed matrix of this transforms p basis to q basis)
    mat = transform_q2p(N)

    # iteration
    t0 = time()
    println("applying channel")
    for t = 1:Nt
        # channel
        map_G1234!(rhot,rhot_p,rhot1,rhot1_p,mat,L,N)
    end
    t1 = time()
    println(round(t1-t0; digits = 3),"sec")

    # plot husimi

    println("preparing for husimi")
    # overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    overlap_qp_mat = load("overlap_qp_mat_L27.jld2", "overlap_qp_mat") # run save_overlap_qp_mat first
    vec_q = [0:1/N:1;] # technically this does not have to be this (I think)
    vec_p = copy(vec_q)
    t2 = time()
    println(round(t2-t1; digits = 3),"sec")
    
    println("calculating husimi")
    f_husimi = husimi(N,rhot,overlap_qp_mat)
    t3 = time()
    println(round(t3-t2; digits = 3),"sec")

    pcolor(vec_q, vec_p, abs.(f_husimi))
    colorbar()

    println("total time:",round(time()-t0; digits = 3),"sec")

end

################### no need to see ###################

function map_G!(rhot,rhot1,L::Int64,N::Int64,ind::Int64)
    # (rhot::Matrix{ComplexF64},rhot1::Matrix{ComplexF64},L::Int64,N::Int64,ind::Int64)

    rhot1 .= 0.0

    for i = 1:L
        for j = 1:L

            ii = 2*L*mod(ind+1,2)+i
            jj = 2*L*mod(ind+1,2)+j

            for n = 0:2
                rhot1[ii,jj] += rhot[mod1(3*ii+n,N),mod1(3*jj+n,N)]
            end
            
        end
    end

end

function transform_q2p_0(N::Int64,psi::Vector{ComplexF64})

    psi_p = zeros(ComplexF64,N)

    for k = 1:N
        for n = 1:N 
            psi_p[k] += exp(1im*2*pi*k*n/N)*psi[n]
        end
    end
    psi_p .= psi_p/sqrt(N)

    return psi_p

end

function transform_q2p_1(N::Int64,rho::Matrix{ComplexF64})

    rho_p = zeros(ComplexF64,N,N)

    for k = 1:N
        for l = 1:N 
            for m = 1:N
                for n = 1:N 
                    rho_p[k,l] += exp(1im*2*pi*(k*m-n*l)/N)*rho[m,n]
                end
            end
        end
    end
    rho_p .= rho_p/N

    return rho_p

end

function transform_q2p(N::Int64)

    mat = zeros(ComplexF64,N,N)

    for k = 1:N
        for n = 1:N 
            mat[k,n] = exp(1im*2*pi*k*n/N)
        end
    end
    mat .= mat/sqrt(N)

    return mat

end

function husimi_overlap(N::Int64,q::Float64,p::Float64)

    overlap_qp = zeros(ComplexF64,N)

    for n = 1:N
        
        for k = 1:N
            braket_kkappa = exp(-pi/N*(k-N/2)^2) * exp(-1im*k*pi)
            for j = 1:N
                overlap_qp[n] += braket_kkappa * exp(1im*2*pi*(-j*n + (j-N*p+N/2)*(k+N*q-N/2))/N)
            end
        end

    end
    overlap_qp .= overlap_qp/N/(N/2)^(1/4)

    return overlap_qp

end

function husimi_overlap_0(N::Int64,rho::Matrix{ComplexF64},q::Float64,p::Float64)

    overlap_qp = zeros(ComplexF64,N)

    for n = 1:N
        
        for k = 1:N
            braket_kkappa = exp(-pi/N*(k-N/2)^2) * exp(-1im*k*pi)
            for j = 1:N
                overlap_qp[n] += braket_kkappa * exp(1im*2*pi*(-j*n + (j-N*p+N/2)*(k+N*q-N/2))/N)
            end
        end

    end
    overlap_qp .= overlap_qp/N/(N/2)^(1/4)

    return overlap_qp'*(rho*overlap_qp)

end

function husim_0(N::Int64,rho::Matrix{ComplexF64})

    vec_q = [0:1/N:1;]
    vec_p = copy(vec_q)
    f_husimi = zeros(ComplexF64,N+1,N+1)
    # overlap_qp_mat = zeros(ComplexF64,N,N+1,N+1)

    for jq = 1:N+1
        for jp = 1:N+1
            f_husimi[jq,jp] = husimi_overlap_0(N,rho,vec_q[jq],vec_p[jp])
            # overlap_qp_mat[:,jq,jp] = husimi_overlap(N,vec_q[jq],vec_p[jp])
        end
    end

    return f_husimi, vec_q, vec_p

end

function husimi_test()

    N = 40

    # reference state
    kappa = zeros(ComplexF64,N)
    for n = 1:N
        kappa[n] = exp(-pi/N*(n-N/2)^2-1im*n*pi)
    end
    kappa .= kappa/(N/2)^(1/4)    

    # rho0 = kappa*kappa'
    psi0 = coherent_state(N,0.75,0.5)
    
    # psi0 .= transform_q2p_0(N,psi0)
    
    rho0 = psi0*psi0'

    rho0 .= transform_q2p(N,rho0)

    # plot husimi
    f_husimi, vec_q, vec_p = husimi(N,rho0)
    pcolor(vec_q, vec_p, abs.(f_husimi))
    colorbar()

end

function transform_q2p_test()

    N = 10

    psi0 = coherent_state(N,0.25,0.25)
    rho0 = psi0*psi0'

    rho1 = transform_q2p_1(N,rho0)

    mat = transform_q2p(N)
    rho2 = mat*rho0*mat'

    return tr(rho1*rho2)

end

function example_14_2()

    L = 3^3 #30 #3^2
    N = 3*L
    Nt = 50

    # define initail state rho0
    # # psi0 = rand(ComplexF64,N)
    # psi0 = zeros(ComplexF64,N)
    # psi0[1] = 1.0
    # psi0 .= psi0/sqrt(sum(abs.(psi0).^2))
    # rho0 = psi0 * psi0'
    # rho0 = rand(Float64,N)
    # rho0 = diagm(rho0)
    # rho0 .= rho0/tr(rho0)
    psi0 = coherent_state(N,0.25,0.25)
    rho0 = psi0*psi0'

    rhot = copy(rho0)
    rhot1 = copy(rhot)
    rhot2 = copy(rhot)

    t0 = time()
    println("applying channel")

    for t = 1:Nt

        # position basis
        map_G!(rhot,rhot1,L,N,1)
        rhot2 .= rhot1*0.25  
        map_G!(rhot,rhot1,L,N,2)
        rhot2 .= rhot2 + rhot1*0.25

        rhot .= transform_q2p(N,rhot)
        map_G!(rhot,rhot1,L,N,1)
        rhot2 .= rhot2 + rhot1*0.25
        map_G!(rhot,rhot1,L,N,2)
        rhot2 .= rhot2 + rhot1*0.25

        # rhot2 .= rhot2*2

        rhot .= rhot2
        
        # if mod(t,1) == 0
        #     println(t)
        # end

        println(round(t/Nt*100; digits = 3),"%")
        t1 = time()
        println(round(t1-t0; digits = 3),"sec")

    end
    t1 = time()

    # plot husimi
    println("preparing for husimi")
    
    overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    t2 = time()
    println(round(t2-t1; digits = 3),"sec")

    # load
    # 
    
    println("calculating husimi")
    f_husimi = husimi(N,rhot,overlap_qp_mat)
    t3 = time()
    println(round(t3-t2; digits = 3),"sec")

    pcolor(vec_q, vec_p, abs.(f_husimi))
    colorbar()

    println("total time:",round(time()-t0; digits = 3),"sec")

    # return rho0, rhot

end

function example_14_1()

    L = 3^5 #30 #3^2
    N = 3*L

    # define initail state rho0
    # # psi0 = rand(ComplexF64,N)
    # psi0 = zeros(ComplexF64,N)
    # psi0[1] = 1.0
    # psi0 .= psi0/sqrt(sum(abs.(psi0).^2))
    # rho0 = psi0 * psi0'
    # rho0 = rand(Float64,N)
    # rho0 = diagm(rho0)
    # rho0 .= rho0/tr(rho0)
    psi0 = coherent_state(N,0.25,0.25)
    rho0 = psi0*psi0'

    rhot = copy(rho0)
    rhot1 = copy(rhot)
    rhot2 = copy(rhot)

    t0 = time()
    println("applying channel")

    for t = 1:10

        map_G!(rhot,rhot1,L,N,1)
        rhot2 .= rhot1*0.25  
        map_G!(rhot,rhot1,L,N,2)
        rhot2 .= rhot2 + rhot1*0.25

        rhot .= transform_q2p(N,rhot)

        map_G!(rhot,rhot1,L,N,1)
        rhot2 .= rhot2 + rhot1*0.25
        map_G!(rhot,rhot1,L,N,2)
        rhot2 .= rhot2 + rhot1*0.25

        rhot .= rhot2
        
        if mod(t,1) == 0
            println(t)
        end

        t1 = time()
        println(t1-t0)

    end

    # plot husimi
    f_husimi, vec_q, vec_p = husimi(N,rhot)
    pcolor(vec_q, vec_p, abs.(f_husimi))
    colorbar()

    # return rho0, rhot

end

function example_14_0()

    L = 3
    N = 3*L

    # define initail state rho0
    # # psi0 = rand(ComplexF64,N)
    # psi0 = zeros(ComplexF64,N)
    # psi0[1] = 1.0
    # psi0 .= psi0/sqrt(sum(abs.(psi0).^2))
    # rho0 = psi0 * psi0'
    rho0 = rand(Float64,N)
    rho0 = diagm(rho0)
    rho0 .= rho0/tr(rho0)

    rhot = copy(rho0)
    rhot1 = copy(rhot)
    choice = rand()

    for t = 1:100000

        # define G1, G2
        if choice < 1/2
            map_G!(rhot,rhot1,L,N,1)
        else
            map_G!(rhot,rhot1,L,N,2)
        end

        rhot .= rhot1
        choice = rand()

        # println(tr(rhot))

    end

    # Husimi...

    return rho0, rhot

end