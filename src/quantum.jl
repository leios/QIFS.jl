using LinearAlgebra
# using QuantumOptics
using PyPlot
# using Plots
using JLD2
# using QuantumToolbox
using BlockDiagonals

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
                # rhot1[2*L+i,2*L+j] += rhot[mod1(3*i1+n,N),mod1(3*j1+n,N)] # which is right, this or below?
                rhot1[i2,j2] += rhot[mod1(3*i2+n,N),mod1(3*j2+n,N)]
                rhot1_p[i1,j1] += rhot_p[mod1(3*i1+n,N),mod1(3*j1+n,N)]
                # rhot1_p[2*L+i,2*L+j] += rhot_p[mod1(3*i1+n,N),mod1(3*j1+n,N)]
                rhot1_p[i2,j2] += rhot_p[mod1(3*i2+n,N),mod1(3*j2+n,N)]
            end

            # for n = 1:3
            #     rhot1[i,j] += rhot[3*(i-1)+n,3*(j-1)+n]
            #     rhot1[i+2*L,j+2*L] += rhot[3*(i-1)+n,3*(j-1)+n]
            #     rhot1_p[i,j] += rhot_p[3*(i-1)+n,3*(j-1)+n]
            #     rhot1_p[i+2*L,j+2*L] += rhot_p[3*(i-1)+n,3*(j-1)+n]
            # end
            
        end
    end

    rhot .= rhot1*0.25 + mat'*rhot1_p*mat*0.25

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

function husimi_format_2(vec_q, vec_p, N::Int64)

    # generate <n|q,p> for all n,q,p for husimi

    # vec_q = [1/N:1/N:1;] #[0:1/N:1;]
    # vec_p = copy(vec_q)
    overlap_qp_mat = zeros(ComplexF64,N,length(vec_q),length(vec_p))
    t0 = time()

    for jq = 1:length(vec_q) #1:N+1 # parallelisable
        for jp = 1:length(vec_p) #1:N+1
            # f_husimi[jq,jp] = husimi_overlap_0(N,rho,vec_q[jq],vec_p[jp])
            overlap_qp_mat[:,jq,jp] = husimi_overlap(N,vec_q[jq],vec_p[jp])
        end

        # show progress
        if mod(jq,10) == 0
            println(round(jq/length(vec_q)*100; digits = 3),"%")
            println(round(time()-t0; digits = 3),"sec")
        end

    end

    return overlap_qp_mat

end

function husimi_format(N::Int64)

    # generate <n|q,p> for all n,q,p for husimi

    vec_q = [1/N:1/N:1;] #[0:1/N:1;]
    vec_p = copy(vec_q)
    overlap_qp_mat = zeros(ComplexF64,N,length(vec_q),length(vec_p))
    t0 = time()

    for jq = 1:length(vec_q) #1:N+1 # parallelisable
        for jp = 1:length(vec_p) #1:N+1
            # f_husimi[jq,jp] = husimi_overlap_0(N,rho,vec_q[jq],vec_p[jp])
            overlap_qp_mat[:,jq,jp] = husimi_overlap(N,vec_q[jq],vec_p[jp])
        end

        # show progress
        if mod(jq,10) == 0
            println(round(jq/length(vec_q)*100; digits = 3),"%")
            println(round(time()-t0; digits = 3),"sec")
        end

    end

    return overlap_qp_mat, vec_q, vec_p

end

function save_overlap_qp_mat()

    # "husimi_format" takes time, and so compute it once and save it

    # L = 3^3
    N = 200 #3*L
    overlap_qp_mat, vec_q, vec_p = husimi_format(N)

    # save
    # jldsave("overlap_qp_mat_L27.jld2"; overlap_qp_mat)
    # jldsave("overlap_qp_mat_N128.jld2"; overlap_qp_mat)
    jldsave("overlap_qp_mat_N200.jld2"; overlap_qp_mat)

end

function husimi(N::Int64,rho::Matrix{ComplexF64},overlap_qp_mat::Array{ComplexF64, 3},vec_q::Vector{Float64},vec_p::Vector{Float64})

    # see Eq.(28) in PRE

    # vec_q = [0:1/N:1;]
    # vec_p = copy(vec_q)

    f_husimi = zeros(ComplexF64,length(vec_q),length(vec_p))

    for jq = 1:length(vec_q)
        for jp = 1:length(vec_p)
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
        println(real(tr(rhot)))
    end
    t1 = time()
    println(round(t1-t0; digits = 3),"sec")

    # plot husimi

    println("preparing for husimi")
    overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    # overlap_qp_mat = load("overlap_qp_mat_L27.jld2", "overlap_qp_mat") # run save_overlap_qp_mat first
    # vec_q = [0:1/N:1;] # technically this does not have to be this (I think)
    # vec_p = copy(vec_q)
    t2 = time()
    println(round(t2-t1; digits = 3),"sec")
    
    println("calculating husimi")
    f_husimi = husimi(N,rhot,overlap_qp_mat,vec_q,vec_p)
    t3 = time()
    println(round(t3-t2; digits = 3),"sec")

    figure()
    pcolor(vec_q, vec_p, abs.(f_husimi))
    colorbar()

    println("total time:",round(time()-t0; digits = 3),"sec")

end

#=-----------------------------------------------------------------------------#
Finite system
#-----------------------------------------------------------------------------=#

function coherent_state_fock(alpha,N)
    psi = zeros(ComplexF64, N)
    psi[1] = 1.0
    for n = 1:N-1
        psi[n+1] = psi[n] * alpha / sqrt(n)
    end
    return psi / norm(psi)
end

function prepation_husimi_fock(q_grid, p_grid, N)
    
    mat = zeros(ComplexF64, N, length(q_grid), length(p_grid))
    
    for j1 = 1:length(q_grid)
        for j2 = 1:length(p_grid)
            alpha = (q_grid[j1] + 1im * p_grid[j2]) / sqrt(2.0)
            mat[:,j1,j2] = coherent_state_fock(alpha, N)
            # reference_state = coherent_state_fock(alpha, N)
            # Q_mat[j1,j2] = real(reference_state'*rho*reference_state)
        end
    end

    return mat

end

function map_from_fock_to_position_or_momentum(N)

    matAdag = zeros(N,N)
    for n = 1:N-1
        matAdag[n+1,n] = sqrt(n)
    end
    matA = matAdag'

    matX = (matAdag+matA)/sqrt(2)
    vals_x, vecs_x = eigen(matX)
    # println(vals)

    # print(vecs'*matX*vecs)

    # return vals/vals[end], vecs, vecs'*matX*vecs

    matP = 1im*(matAdag-matA)/sqrt(2)
    vals_p, vecs_p = eigen(matP)
    # println(vals_p)

    return vecs_x, vecs_p, vals_x

end

function channel_square_expand!(rhot,rhot_X,rhot_P,rhot1_X,rhot1_P,matFtoX,matFtoP,M,L,N,t,Nt)

    rhot_X .= matFtoX'*rhot*matFtoX
    rhot_P .= matFtoP'*rhot*matFtoP

    rhot1_X .= 0.0
    rhot1_P .= 0.0
    
    for i = 1:L
        for j = 1:L

            i1 = i + M
            j1 = j + M
            i2 = i + M + L
            j2 = j + M + L
            
            for n = 1:Int64(N/L)
                rhot1_X[i1,j1] += rhot_X[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
                rhot1_X[i2,j2] += rhot_X[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
                rhot1_P[i1,j1] += rhot_P[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
                rhot1_P[i2,j2] += rhot_P[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
            end

        end
    end
    
    if t == Nt
        rhot .= (matFtoX*rhot1_X*matFtoX')*0.25 + (matFtoP*rhot1_P*matFtoP')*0.25
    else
        rhot_X .= 0.0
        rhot_P .= 0.0
        for i = 1:L
            for j = 1:L

                i1 = i + M
                j1 = j + M
                i2 = i + M + L
                j2 = j + M + L
                
                for n = 1:Int64(N/L)
                    rhot_X[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n] += rhot1_X[i1,j1]/Int64(N/L)
                    rhot_X[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n] += rhot1_X[i2,j2]/Int64(N/L)
                    rhot_P[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n] += rhot1_P[i1,j1]/Int64(N/L)
                    rhot_P[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n] += rhot1_P[i2,j2]/Int64(N/L)
                end

            end
        end
        rhot .= (matFtoX*rhot_X*matFtoX')*0.25 + (matFtoP*rhot_P*matFtoP')*0.25
    end

end

function rotation(N,theta)

    matN = diagm(0 => 0:N-1)*1.0
    matR = exp(1im*theta*matN)

    return matR

end

function displacement(N,alpha)

    matAdag = zeros(N,N)
    for n = 1:N-1
        matAdag[n+1,n] = sqrt(n)
    end
    matA = matAdag'

    matD = (alpha*matAdag - conj(alpha)*matA)/sqrt(2)
    matD .= exp(matD)

    return matD

end

function shearing(N,beta)

    matAdag = zeros(N,N)
    for n = 1:N-1
        matAdag[n+1,n] = sqrt(n)
    end
    matA = matAdag'

    matX = (matAdag+matA)/sqrt(2)
    matP = 1im*(matAdag-matA)/sqrt(2)

    matS = exp(1im*beta*matX*matX)
    # matS = exp(1im*beta*matX^3)
    # matS = exp(1im*beta*matX*matP)

    return matS

end

function log_binomial(a::Int64,b::Int64)

    result_denominator = 0.0
    result_numerator = 0.0
    b1 = min(b,a-b)

    for n = 1:b1
        result_denominator += log(n)
        result_numerator += log(a-n-1)
    end

    return result_numerator - result_denominator

end

function expansion(N,lambda,rho)

    rho1 = zeros(ComplexF64, N, N)
    
    for k = 0:N-1
        for s = 0:N-1
            for j = 0:min(N-1-k,N-1-s)
                log_coeff_ket = log_binomial(k+j,k)/2 + (k+1)*log(lambda) + (j/2)*log(1-lambda^2)
                log_coeff_bra = log_binomial(s+j,s)/2 + (s+1)*log(lambda) + (j/2)*log(1-lambda^2)
                rho1[k+j+1,s+j+1] += exp(log_coeff_ket)*exp(log_coeff_bra)*rho[k+1,s+1]
                # coeff_ket = sqrt(binomial(k+j,k))*lambda^(k+1)*sqrt(1-lambda^2)^j
                # coeff_bra = sqrt(binomial(s+j,s))*lambda^(s+1)*sqrt(1-lambda^2)^j
                # rho1[k+j+1,s+j+1] += coeff_ket*coeff_bra*rho[k+1,s+1]
            end
        end
    end

    if real(tr(rho1)) < 0.98
        println("scaling parameter is so big or your system is so small that some states excape from your finite system")
    end

    return rho1

end

function get_superoperator_expansion()

    # super operator for expansion

    N = 2^5
    mat = zeros(ComplexF64, N^2, N^2)

    for k = 0:N-1
        for s = 0:N-1

            ind1 = k+1 + (s+1)*N # wrong

            for j = 0:min(N-1-k,N-1-s)

                ind2 = k+j+1 + (s+j+1)*N

                log_coeff_ket = log_binomial(k+j,k)/2 + (k+1)*log(lambda) + (j/2)*log(1-lambda^2)
                log_coeff_bra = log_binomial(s+j,s)/2 + (s+1)*log(lambda) + (j/2)*log(1-lambda^2)

                mat[ind2,ind1] += exp(log_coeff_ket)*exp(log_coeff_bra)

            end
        end
    end

    return mat

end

function get_kraus_expansion()

    mat = get_superoperator_expansion()

end


function squeezing(N,chi)

    # chi is complex, chi = r exp(i*theta) with r suqeeing strength and theta suqeezing angle

    matAdag = zeros(N,N)
    for n = 1:N-1
        matAdag[n+1,n] = sqrt(n)
    end
    matA = matAdag'

    matS = (conj(chi)*matA^2 - chi*matAdag^2)/2
    matS .= exp(matS)

    return matS

end

function division_X(rhot,matFtoX,matFtoP,L,N)

    # (x,y) -> (x/lambda, y*lambda) with lambda = N/L (integer)

    rhot_X = matFtoX'*rhot*matFtoX
    rhot1_X = copy(rhot_X)
    rhot1_X .= 0.0
    
    for i = 1:L
        for j = 1:L            
            for n = 1:Int64(N/L)
                rhot1_X[i,j] += rhot_X[Int64(N/L)*(i-1)+n, Int64(N/L)*(j-1)+n]
            end
        end
    end

    # return matFtoX*rhot1_X*matFtoX'

    # translation
    matV = zeros(N,N)
    for i = 1:N
        matV[mod1(i+1,N),i] = 1
    end

    rhot1_P = matFtoP'*(matFtoX*rhot1_X*matFtoX')*matFtoP
    for j = 1:Int64(N/2)
        rhot1_P .= matV*rhot1_P*matV'
    end

    return matFtoP*rhot1_P*matFtoP'

end

function map_division_0!(rhot::Matrix{ComplexF64},rhot_p::Matrix{ComplexF64},rhot1::Matrix{ComplexF64},rhot1_p::Matrix{ComplexF64},mat::Matrix{ComplexF64},M::Int64,L::Int64,N::Int64,t::Int64,Nt::Int64)

    rhot_p .= mat*rhot*mat' # rhot in p basis
    rhot1 .= 0.0
    rhot1_p .= 0.0

    for i = 1:L
        for j = 1:L
            for n = 1:Int64(N/L)
                rhot1[i,j] += rhot[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
            end
        end
    end

    # translation
    matV = zeros(N,N)
    for i = 1:N
        matV[mod1(i+1,N),i] = 1
    end

    rhot1_p .= mat*rhot1*mat'
    for j = 1:Int64(N/2)
        rhot1_p .= matV*rhot1_p*matV'
    end
    rhot1 .= mat'*rhot1_p*mat


    rhot .= rhot1

end

function example_fock()

    N = 200
    M = Int64(N/4)
    L = Int64(N/4)
    
    Nt = 0 # number of iteration

    # maping from fock basis to position or momentum basis
    matFtoX, matFtoP, array_x = map_from_fock_to_position_or_momentum(N)

    # define initail state rho0
    psi0 = coherent_state_fock((0.0+1im*0)/sqrt(2),N) # (q+1im*p)/sqrt(2)
    rho0 = psi0*psi0'
    # rho0 = zeros(ComplexF64, N, N)
    # rho0[1,1] = 1.0

    #
    rho0 .= 0.0
    # rho0[10,10] = 1.0
    for n = 1:10
        rho0[n,n] = 1.0
    end
    # rho0[5,1] = 1
    rho0 .= rho0/tr(rho0)
    # psi0 .= psi0/norm(psi0)
    # rho0 = psi0*psi0'
    # rho0 = Matrix{ComplexF64}(I, N, N)/N

    # ancilla = [1,1]/sqrt(2)
    # ancilla = ancilla*ancilla'
    # rho0 = kron(ancilla,rho0)

    # define density matrices
    rhot = copy(rho0) # time-evolved state to track
    rhot_X = copy(rhot)
    rhot_P = copy(rhot)
    rhot1_X = copy(rhot)
    rhot1_P = copy(rhot)
    
    # measure time
    t0 = time()
    println(tr(rhot))

    # iteration for a primative
    for t = 1:Nt
        # channel
        channel_square_expand!(rhot,rhot_X,rhot_P,rhot1_X,rhot1_P,matFtoX,matFtoP,M,L,N,t,Nt)
        println(tr(rhot))
    end

    # rotation
    theta = pi/8
    matR = rotation(N,theta)
    # rhot .= matR*rhot*matR'

    # translation
    alpha = 5.0
    matD = displacement(N,alpha)
    # rhot .= matD*rhot*matD'

    # shearing
    beta = 0.2
    matS = shearing(N,beta)
    # rhot .= matS*rhot*matS'

    # expansion
    lambda = 0.5 # needs to be smaller than 1
    # rhot .= expansion(N,lambda,rhot)
    
    # squeezing
    chi = 1.0
    matSq = squeezing(N,chi)
    # rhot .= matSq*rhot*matSq'

    # division (not working but the same thing works in a periodic system; see example_square_decoupled and map_division!)
    # L0 = Int64(N/2)
    # rhot .= division_X(rhot,matFtoX,matFtoP,L0,N)

    #
    # matI = [1.0,0]
    # matD2 = kron(matD,matD)
    # rhot = matD2'*rhot*matD

    # plot husimi
    println(tr(rhot))
    println("preparing for husimi")

    q_range = copy(array_x) 
    # q_range = collect(range(0.0, 10.0, length=N))
    p_range = copy(q_range)
    mat_husimi = prepation_husimi_fock(q_range, p_range, N)
    f_husimi = husimi(N,rhot,mat_husimi,q_range,p_range)

    figure()
    pcolor(q_range, p_range, abs.(f_husimi)')
    colorbar()

    # figure()
    # plot(abs.(f_husimi[Int64(N/2),:]))

    println("total time:",round(time()-t0; digits = 3),"sec")


end

function example_fock_duplication()

    N = 2^5 #2^6

    # maping from fock basis to position or momentum basis
    matFtoX, matFtoP, array_x = map_from_fock_to_position_or_momentum(N)

    # return array_x

    # define initail state rho0
    rho0 = zeros(ComplexF64, N, N)
    for n = 1:2^3 #2^3
        rho0[n,n] = 1.0
    end
    rho0 .= rho0/tr(rho0)

    ancilla = [1,1]/sqrt(2)
    ancilla = ancilla*ancilla'
    rho0 = kron(ancilla,rho0)

    # return rho0

    #
    alpha = 4.0
    matD2 = BlockDiagonal([displacement(N,-alpha),displacement(N,alpha)])
    rhot = matD2*rho0*matD2'

    # return rhot

    rhot_ptraceout = zeros(ComplexF64, N, N)

    for i = 1:N
        for j = 1:N
            rhot_ptraceout[i,j] = rhot[i,j] + rhot[i+N,j+N]
        end
    end
    rhot = rhot_ptraceout

    # plot husimi
    println(tr(rhot))
    println("preparing for husimi")

    q_range = copy(array_x) 
    # q_range = collect(range(0.0, 10.0, length=N))
    p_range = copy(q_range)
    mat_husimi = prepation_husimi_fock(q_range, p_range, N)
    f_husimi = husimi(N,rhot,mat_husimi,q_range,p_range)

    figure()
    pcolor(q_range, p_range, abs.(f_husimi)')
    colorbar()

    # figure()
    # plot(abs.(f_husimi[Int64(N/2),:]))

end

#=-----------------------------------------------------------------------------#
Shearing
#-----------------------------------------------------------------------------=#

function example_shearing()

    N = 40 # system size
    L = Int64(N/2)

    Nt = 1 # number of iteration

    # define matrix to transform q basis to p basis (complex conjugate transposed matrix of this transforms p basis to q basis)
    matN = transform_q2p(N)

    # define initail state rho0
    psi0 = coherent_state(N,0.5,0.5) # can be any state in principle but we choose a cherent state
    #psi0 .= 0.0
    #psi0[L]=1
    # psi0 .= matN'*psi0
    rho0 = psi0*psi0'

    # define density matrices
    rhot = copy(rho0) # time-evolved state to track
    
    # measure time
    t0 = time()

    # iteration
    beta = 1.5 # shearing parameter
    map_shearing = zeros(ComplexF64,N,N)
    for n = 1:N
        map_shearing[n,n] = exp(1im*2*pi/N*beta*n*n) # relocate this with amonut the non-integer causes
    end
    for t = 1:Nt
        rhot .= map_shearing*rhot*map_shearing'
    end
    
    # plot husimi

    println("preparing for husimi")
    overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    # overlap_qp_mat = load("overlap_qp_mat_N128.jld2", "overlap_qp_mat") # run save_overlap_qp_mat first
    # vec_q = [0:1/N:1;] # technically this does not have to be this (I think)
    # vec_p = copy(vec_q)
    
    figure()
    f_husimi = husimi(N,rhot,overlap_qp_mat,vec_q,vec_p)
    pcolor(vec_q, vec_p, abs.(f_husimi)')
    colorbar()

    println("total time:",round(time()-t0; digits = 3),"sec")

end

#=-----------------------------------------------------------------------------#
Rotation
#-----------------------------------------------------------------------------=#

function test_rotation(N)

    if mod(N,4) ~= 2
        error("N needs to be 4m+2 for now.")
    end

    mat = zeros(ComplexF64,N,N)

    for k = 1:N
        for n = 1:N 
            mat[k,n] = exp(1im*2*pi*k*n/N)
        end
    end
    mat .= mat/sqrt(N)

    #vals, vecs = eigen(mat)
    #println(vals)

    mat_S = zeros(ComplexF64,N,N)

    for k = 1:N
        mat_S[k,k] = 2*cos(2*pi*k/N)
        if k-1 >= 1
            mat_S[k,k-1] = 1
        end
        if k+1 <= N
            mat_S[k,k+1] = 1
        end
    end
    mat_S[1,N] = 1
    mat_S[N,1] = 1

    #println(norm(mat*mat_S-mat_S*mat))

    vals_S, vecs_S = eigen(mat_S)

    # array_d = zeros(ComplexF64,N)
    # for n =1:N
    #     array_d[n] = exp(1im*pi/2*n)
    # end

    # array_d = zeros(ComplexF64,N)
    # array_d[1] = 1.0
    # for n =2:Int64(N/2)
    #     array_d[n] = array_d[n-1]*exp(1im*pi/2)
    # end
    # array_d[Int64(N/2)+1] = array_d[Int64(N/2)]*exp(1im*pi)
    # for n = Int64(N/2)+2:N
    #     array_d[n] = array_d[n-1]*exp(-1im*pi/2)
    # end
    # array_d .= array_d*exp(1im*pi)

    alpha = 0.8

    array_d = zeros(ComplexF64,N)
    array_d[1] = 1.0
    for n =2:Int64(N/2)
        array_d[n] = array_d[n-1]*exp(1im*pi/2*alpha)
    end
    array_d[Int64(N/2)+1] = array_d[Int64(N/2)]*exp(1im*pi*alpha)
    for n = Int64(N/2)+2:N
        array_d[n] = array_d[n-1]*exp(-1im*pi/2*alpha)
    end
    
    # return diag(transpose(vecs_S)*mat*vecs_S), array_d*exp(1im*pi/2)

    array_d_true = diag(transpose(vecs_S)*mat*vecs_S)

    # mat_R = vecs_S*diagm(array_d*exp(1im*pi/2))*transpose(vecs_S)
    mat_R = vecs_S*diagm(array_d)*transpose(vecs_S)

    # return diagm(vals)

    # return mat, vecs*diagm(vals)*inv(vecs)

    # return transpose(vecs_S)*mat*vecs_S, transpose(vecs_S)*mat_R*vecs_S

    return mat_R

    # return array_d, array_d_true, vals_S

end

function map_rotation!(rhot,rhot1,rhot_p,rhot1_p,mat,L,N)

    # vals, vecs = eigen(mat)
    # sqrt_vals = sqrt.(vals)
    # sqrt_mat = vecs*diagm(sqrt_vals)*inv(vecs)

    # map_rotation = zeros(ComplexF64,N,N)
    # for n = 1:N
    #     map_rotation[n,n] = exp(1im*2*pi/N*n)
    # end

    # mat_x = zeros(ComplexF64,N,N)
    # for k = 1:N
    #     for n = 1:N 
    #         if k == n
    #             mat_x[k,n] = n/N
    #         end
    #     end
    # end

    # mat_p = copy(mat_x)
    # mat_p .= mat'*mat_p*mat
    
    # mat_x2 = mat_x^2
    # mat_p2 = mat_p^2

    # map_rotation = zeros(ComplexF64,N,N)
    # for k = 1:N
    #     for n = 1:N 
    #         # theta = 
    #         map_rotation[k,n] = exp(-1im*(mat_x2[k,n]+mat_p2[k,n])/2*pi)
    #     end
    # end
    # map_rotation .= map_rotation/sqrt(N)

    map_rotation = test_rotation(N)

    rhot .= map_rotation*rhot*map_rotation'

end

function example_rotation()

    N = 42 #40 # system size
    Nt = 1 # number of iteration
    L = Int64(N/2)

    # define matrix to transform q basis to p basis (complex conjugate transposed matrix of this transforms p basis to q basis)
    mat = transform_q2p(N)

    # define initail state rho0
    psi0 = coherent_state(N,1/2,1/2) # can be any state in principle but we choose a cherent state
    # psi0 = coherent_state(N,1/2,1/3)+coherent_state(N,1/2,5/6)
    # psi0 .= psi0/sqrt(sum(abs.(psi0).^2))
    psi0 .= 0.0
    psi0[L]=1
    # psi0 .= mat'*psi0
    rho0 = psi0*psi0'

    # define density matrices
    rhot = copy(rho0) # time-evolved state to track
    rhot_p = copy(rhot)
    rhot1 = copy(rhot)
    rhot1_p = copy(rhot)

    # translation
    matV = zeros(N,N)
    for i = 1:N
        matV[mod1(i+1,N),i] = 1
    end
    
    # measure time
    t0 = time()

    # iteration
    for t = 1:Nt
        # channel
        
        map_rotation!(rhot,rhot1,rhot_p,rhot1_p,mat,L,N)

    end
    
    # plot husimi

    println("preparing for husimi")
    overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    
    figure()
    f_husimi = husimi(N,rhot,overlap_qp_mat,vec_q,vec_p)
    pcolor(vec_q, vec_p, abs.(f_husimi)')
    colorbar()

    println("total time:",round(time()-t0; digits = 3),"sec")

end

#=-----------------------------------------------------------------------------#
Scale
#-----------------------------------------------------------------------------=#

function generate_kraus_case_a(lambda,N)
    
    K_all = [zeros(Float64, N, N) for k in 1:N]

    for k = 0:(N-1)
        for n = 0:(N-k-1)
            K_all[k+1][n+k+1, n+1] = sqrt(binomial(n+k, k)) * lambda^(n+k+1) * (1 - lambda^2)^(k/2)
        end
    end
    
    # for m = 0:(N-1)
    #     for n = m:N-1
    #         K_all[m+1][n-m+1, n+1] = sqrt(binomial(n, m)) * (lambda^m) * (1 - lambda^2)^((n-m)/2)
    #     end
    # end

    return K_all
end

function map_case_a(lambda,N,rho)

    rho1 = zeros(ComplexF64, N, N)
    
    for k = 0:N-1
        for s = 0:N-1
            for j = 0:min(N-1-k,N-1-s)
                coeff_ket = sqrt(binomial(k+j,k))*lambda^(k+1)*sqrt(1-lambda^2)^j
                coeff_bra = sqrt(binomial(s+j,s))*lambda^(s+1)*sqrt(1-lambda^2)^j
                # rho1[k+j+1,s+j+1] += coeff_ket*coeff_bra*psi[k+1]*psi[s+1]'
                rho1[k+j+1,s+j+1] += coeff_ket*coeff_bra*rho[k+1,s+1]
            end
        end
    end

    return rho1

    # rho_lambda = zeros(N,N)
    # for j = 0:N-1
    #     rho_lambda[j+1,j+1] = lambda^2*(1-lambda^2)^j
    # end

end

# function coherent_state_fock(alpha,N)
#     psi = zeros(ComplexF64, N)
#     psi[1] = 1.0
#     for n in 1:(N-1)
#         psi[n+1] = psi[n] * alpha / sqrt(n)
#     end
#     return psi / norm(psi)
# end

function husimi_fock(rho, q_grid, p_grid, N)
    
    Q_mat = zeros(Float64, length(q_grid), length(p_grid))
    
    for j1 = 1:length(q_grid)
        for j2 = 1:length(p_grid)
            alpha = (q_grid[j1] + 1im * p_grid[j2]) / sqrt(2.0)
            reference_state = coherent_state_fock(alpha, N)
            Q_mat[j1,j2] = real(reference_state'*rho*reference_state)
        end
    end

    return Q_mat

end

function test_scale()

    N = 40
    lambda = 0.8

    rho_0 = zeros(ComplexF64, N, N)
    # rho_lambda = zeros(ComplexF64, N, N)

    rho_0[1,1] = 1.0
    
    # psi_0 = zeros(ComplexF64, N)
    # psi_0[1] = 1.0
    
    # Kraus_ops = generate_kraus_case_a(lambda, N)

    # completeness_sum = zeros(Float64, N, N)
    # for Km in Kraus_ops
    #     completeness_sum += Km' * Km
    # end
    # return Kraus_ops
    # return completeness_sum
    # println(completeness_sum)

    # for Km in Kraus_ops
    #     rho_lambda += Km * rho_0 * Km'
    # end

    #
    # rho_lambda0 = zeros(N,N)
    # for j = 0:N-1
    #     rho_lambda0[j+1,j+1] = lambda^2*(1-lambda^2)^j
    # end

    #
    # rho_lambda = map_case_a(lambda,N,rho_0)
    # println(tr(rho_lambda))
    # return rho_0, rho_lambda0, rho_lambda

    rho_lambda = copy(rho_0)
    for t=1:4
        rho_lambda .= map_case_a(lambda,N,rho_lambda)
        # rho_lambda .= expansion(N,lambda,rho_lambda)
        println(tr(rho_lambda))
    end


    q_range = -5:0.2:5
    p_range = -5:0.2:5

    Q_initial = husimi_fock(rho_0, q_range, p_range, N)
    Q_contracted = husimi_fock(rho_lambda, q_range, p_range, N)

    figure()
    pcolor(q_range, p_range, abs.(Q_initial))
    colorbar()

    figure()
    pcolor(q_range, p_range, abs.(Q_contracted))
    colorbar()

    # N = 10
    # b_fock = FockBasis(N)

    # vac_state = fockstate(b_fock, 0)
    
    # rho_vac = dm(vac_state)

    # #


    # xvec = -5:0.2:5
    # yvec = -5:0.2:5
    # Q_values = qfunc(rho_vac, xvec, yvec)

    # figure()
    # pcolor(xvec, yvec, abs.(Q_values))
    # colorbar()

end

#=-----------------------------------------------------------------------------#
Compresion from two different directions
#-----------------------------------------------------------------------------=#

function map_translation!(rhot,rhot1,rhot_p,rhot1_p,mat,L,N)

    rhot_p .= mat*rhot*mat' # rhot in p basis
    rhot1 .= 0.0
    rhot1_p .= 0.0

    matV = zeros(N,N)
    for i = 1:N
        matV[mod1(i+1,N),i] = 1
    end

    rho_t .= matV*rhot*matV'

    # rhot1_p .= matV*rhot_p*matV'
    # rhot .= mat'*rhot1_p*mat

end

function map_compression_test!(rhot,rhot1,rhot_p,rhot1_p,mat,L,N,t,Nt)

    rhot_p .= mat*rhot*mat' # rhot in p basis
    rhot1 .= 0.0
    rhot1_p .= 0.0

    for i = 1:L
        for j = 1:L

            i1 = L+i
            j1 = L+j
            i2 = 2*L+i
            j2 = 2*L+j

            for n = 0:3
                rhot1[i1,j1] += rhot[mod1(4*i1+n,N),mod1(4*j1+n,N)]
                rhot1[i2,j2] += rhot[mod1(4*i2+n,N),mod1(4*j2+n,N)]
                # rhot1_p[i1,j1] += rhot_p[mod1(4*i1+n,N),mod1(4*j1+n,N)]
                # rhot1_p[i2,j2] += rhot_p[mod1(4*i2+n,N),mod1(4*j2+n,N)]
            end
            
        end
    end

    rhot .= rhot1/2
    # rhot .= rhot1*0.25 + mat'rhot1_p*mat*0.25

    if t == Nt
        rhot .= rhot1*0.25*2# + mat'rhot1_p*mat*0.25
    else
        rhot .= 0.0
        rhot_p .= 0.0
        for i = 1:L
            for j = 1:L
                i1 = i + L
                j1 = j + L
                i2 = i + L*2
                j2 = j + L*2
                for n = 0:3
                    rhot[mod1(4*i1+n,N),mod1(4*j1+n,N)] += rhot1[i1,j1]/4
                    rhot[mod1(4*i2+n,N),mod1(4*j2+n,N)] += rhot1[i2,j2]/4
                    # rhot_p[mod1(4*i1+n,N),mod1(4*j1+n,N)] += rhot1_p[i1,j1]/4
                    # rhot_p[mod1(4*i2+n,N),mod1(4*j2+n,N)] += rhot1_p[i2,j2]/4
                end            
            end
        end
        # println(tr(rhot))
        rhot .= rhot*0.25*2# + mat'*rhot_p*mat*0.25
    end

end

function map_compression!(rhot,rhot1,rhot_p,rhot1_p,mat,L,N)

    # rhot_p .= mat*rhot*mat' # rhot in p basis
    rhot1 .= 0.0
    # rhot1_p .= 0.0

    for i = 1:L
        for j = 1:L

            i1 = i
            j1 = j
            # i2 = L+i
            # j2 = L+j

            for n = 0:1
                rhot1[i1,j1] += rhot[mod1(2*i1+n,N),mod1(2*j1+n,N)]
            end
            
        end
    end

    rhot .= rhot1

end

function map_compression_p!(rhot,rhot1,rhot_p,rhot1_p,mat,L,N)

    rhot_p .= mat*rhot*mat' # rhot in p basis
    rhot1 .= 0.0
    rhot1_p .= 0.0

    for i = 1:L
        for j = 1:L

            i1 = i
            j1 = j
            # i2 = L+i
            # j2 = L+j

            for n = 0:1
                rhot1_p[i1,j1] += rhot_p[mod1(2*i1+n,N),mod1(2*j1+n,N)]
            end
            
        end
    end

    rhot .= mat'*rhot1_p*mat

end

function test_amplification(N=40,lambda=1.0,Nk=20)

    # N = 5

    # lambda = 1.1
    # Nk = 20

    matK = zeros(N,N,Nk)
    
    for n = 1:N
        matK[n,n,1] += 1/sqrt(lambda^(n-1))
    end
    matK[:,:,1] .= matK[:,:,1]/sqrt(lambda)

    for k = 2:Nk
        for n = 1:N
            matK[mod1(n+(k-1),N),n,k] += sqrt(binomial(n+k-2,k-1))/sqrt(lambda^(n-1))
        end
        matK[:,:,k] .= matK[:,:,k]/sqrt(lambda)*sqrt(1-1/lambda)^(k-1)
    end

    mat = zeros(N,N,Nk)
    mat1 = zeros(N,N)
    for k = 1:Nk
        mat[:,:,k] = matK[:,:,k]'*matK[:,:,k]
        mat1 += matK[:,:,k]'*matK[:,:,k]
    end
    
    return matK, mat, mat1

    # return matK

end

function test_creationbasis()

    N = 40

    matAdag = zeros(N,N)
    for n = 1:N-1
        matAdag[n+1,n] = sqrt(n)
    end
    matA = matAdag'

    matX = (matAdag+matA)/sqrt(2)
    vals, vecs = eigen(matX)

    # print(vecs'*matX*vecs)

    # return vals/vals[end], vecs, vecs'*matX*vecs

    return vecs

end

function husimi_format_expand(N::Int64)

    vec_q = [1/N:1/N:1;] #[0:1/N:1;]
    vec_p = copy(vec_q)
    overlap_qp_mat = zeros(ComplexF64,N,length(vec_q),length(vec_p))
    t0 = time()

    for jq = 1:length(vec_q) #1:N+1 # parallelisable
        for jp = 1:length(vec_p) #1:N+1
            # f_husimi[jq,jp] = husimi_overlap_0(N,rho,vec_q[jq],vec_p[jp])
            overlap_qp_mat[:,jq,jp] = husimi_overlap(N,vec_q[jq]/2,vec_p[jp]/2)
        end

        # show progress
        if mod(jq,10) == 0
            println(round(jq/length(vec_q)*100; digits = 3),"%")
            println(round(time()-t0; digits = 3),"sec")
        end

    end

    return overlap_qp_mat, vec_q, vec_p

end

function example_compression()

    N = 40 # system size
    Nt = 4 # number of iteration
    # L = Int64(N/4)
    L = Int64(N/2)

    # define matrix to transform q basis to p basis (complex conjugate transposed matrix of this transforms p basis to q basis)
    mat = transform_q2p(N)

    # define initail state rho0
    psi0 = coherent_state(N,1/2,1/2) # can be any state in principle but we choose a cherent state
    # psi0 = coherent_state(N,1/2,1/3)+coherent_state(N,1/2,5/6)
    # psi0 .= psi0/sqrt(sum(abs.(psi0).^2))
    # psi0 .= 0.0
    # psi0[L]=1
    # psi0 .= mat'*psi0
    rho0 = psi0*psi0'

    # define density matrices
    rhot = copy(rho0) # time-evolved state to track
    rhot_p = copy(rhot)
    rhot1 = copy(rhot)
    rhot1_p = copy(rhot)

    # translation
    matV = zeros(N,N)
    for i = 1:N
        matV[mod1(i+1,N),i] = 1
    end

    # dilatation
    # matX = diagm(0 => 1:N)*(1.0+1im*0.0)
    # matP = diagm(0 => 1:N)*(1.0+1im*0.0)
    # matP .= mat'*matP*mat
    # matD = (matX*matP + matP*matX)/2

    # alpha = 1.05
    # dilatation = exp(-1im*log(alpha)*matD)

    lambda = 1.1
    Nk = 20
    matK, mat, mat1 = test_amplification(N,lambda,Nk)
    
    matP = test_creationbasis()
    
    # measure time
    t0 = time()

    # iteration
    for t = 1:Nt
        
        # channel 
        rhot1 .= 0.0
        for k = 1:Nk
            rhot1 += matK[:,:,k]*(matP*rhot*matP')*matK[:,:,k]'
        end
        rhot .= matP'*rhot1*matP

        # rhot .= dilatation*rhot*dilatation'

        # map_compression_test!(rhot,rhot1,rhot_p,rhot1_p,mat,L,N,t,Nt)

        # map_compression!(rhot,rhot1,rhot_p,rhot1_p,mat,L,N)
        # for t1 = 1:L
        #     rhot .= mat'*(matV*(mat*rhot*mat')*matV')*mat
        # end

        # map_compression_p!(rhot,rhot1,rhot_p,rhot1_p,mat,L,N)
        # for t1 = 1:Int64(L/2)
        #     rhot .= matV'*rhot*matV
        # end

        println(tr(rhot))

    end
    
    # plot husimi

    println("preparing for husimi")
    overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    # overlap_qp_mat, vec_q, vec_p = husimi_format_expand(N)
    
    figure()
    f_husimi = husimi(N,rhot,overlap_qp_mat,vec_q,vec_p)
    pcolor(vec_q, vec_p, abs.(f_husimi)')
    colorbar()

    println("total time:",round(time()-t0; digits = 3),"sec")

end

#=-----------------------------------------------------------------------------#
Scale
#-----------------------------------------------------------------------------=#

function map_expand!(rhot::Matrix{ComplexF64},rhot_p::Matrix{ComplexF64},rhot1::Matrix{ComplexF64},rhot1_p::Matrix{ComplexF64},mat::Matrix{ComplexF64},M::Int64,L::Int64,N::Int64,t::Int64,Nt::Int64)

    rhot_p .= mat*rhot*mat' # rhot in p basis
    rhot1 .= 0.0
    rhot1_p .= 0.0

    for i = 1:L
        for j = 1:L
            i1 = i + M
            j1 = j + M
            i2 = i + M + L
            j2 = j + M + L
            for n = 0:3  ########### check
                rhot1[i1,j1] += rhot[mod1(4*i1+n,N),mod1(4*j1+n,N)] ########### check
                rhot1[i2,j2] += rhot[mod1(4*i2+n,N),mod1(4*j2+n,N)]
                rhot1_p[i1,j1] += rhot_p[mod1(4*i1+n,N),mod1(4*j1+n,N)]
                rhot1_p[i2,j2] += rhot_p[mod1(4*i2+n,N),mod1(4*j2+n,N)]
            end            
        end
    end
    
    if t == Nt
        rhot .= rhot1*0.25 + mat'*rhot1_p*mat*0.25
    else
        rhot .= 0.0
        rhot_p .= 0.0
        for i = 1:L
            for j = 1:L
                i1 = i + M
                j1 = j + M
                i2 = i + M + L
                j2 = j + M + L
                for n = 0:3  ########### check
                    rhot[mod1(4*i1+n,N),mod1(4*j1+n,N)] += rhot1[i1,j1]/4 ########### check
                    rhot[mod1(4*i2+n,N),mod1(4*j2+n,N)] += rhot1[i2,j2]/4
                    rhot_p[mod1(4*i1+n,N),mod1(4*j1+n,N)] += rhot1_p[i1,j1]/4
                    rhot_p[mod1(4*i2+n,N),mod1(4*j2+n,N)] += rhot1_p[i2,j2]/4
                end            
            end
        end
        # println(tr(rhot))
        rhot .= rhot*0.25 + mat'*rhot_p*mat*0.25
    end

end

#=-----------------------------------------------------------------------------#
SQUARE
#-----------------------------------------------------------------------------=#

function map_division!(rhot::Matrix{ComplexF64},rhot_p::Matrix{ComplexF64},rhot1::Matrix{ComplexF64},rhot1_p::Matrix{ComplexF64},mat::Matrix{ComplexF64},M::Int64,L::Int64,N::Int64,t::Int64,Nt::Int64)

    rhot_p .= mat*rhot*mat' # rhot in p basis
    rhot1 .= 0.0
    rhot1_p .= 0.0

    for i = 1:L
        for j = 1:L
            # i1 = i + M
            # j1 = j + M
            # i2 = i + M + L
            # j2 = j + M + L
            
            # # for n = 0:3  
            # for n = 0:Int64(N/L)-1 ########### check
            #     # rhot1[i1,j1] += rhot[mod1(4*i1+n,N),mod1(4*j1+n,N)]
            #     # rhot1[i2,j2] += rhot[mod1(4*i2+n,N),mod1(4*j2+n,N)]
            #     # rhot1_p[i1,j1] += rhot_p[mod1(4*i1+n,N),mod1(4*j1+n,N)]
            #     # rhot1_p[i2,j2] += rhot_p[mod1(4*i2+n,N),mod1(4*j2+n,N)]
            #     # rhot1[i1,j1] += rhot[mod1(Int64(N/L)*i1+n,N),mod1(Int64(N/L)*j1+n,N)] ########### check
            #     # rhot1[i2,j2] += rhot[mod1(Int64(N/L)*i2+n,N),mod1(Int64(N/L)*j2+n,N)]
            #     # rhot1_p[i1,j1] += rhot_p[mod1(Int64(N/L)*i1+n,N),mod1(Int64(N/L)*j1+n,N)]
            #     # rhot1_p[i2,j2] += rhot_p[mod1(Int64(N/L)*i2+n,N),mod1(Int64(N/L)*j2+n,N)]
            #     rhot1[i1,j1] += rhot[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)] ########### check
            #     rhot1[i2,j2] += rhot[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)]
            #     rhot1_p[i1,j1] += rhot_p[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)]
            #     rhot1_p[i2,j2] += rhot_p[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)]
            # end

            for n = 1:Int64(N/L)
                rhot1[i,j] += rhot[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
                # rhot1[i2,j2] += rhot[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
                # rhot1_p[i1,j1] += rhot_p[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
                # rhot1_p[i2,j2] += rhot_p[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
            end

        end
    end

     # translation
    matV = zeros(N,N)
    for i = 1:N
        matV[mod1(i+1,N),i] = 1
    end

    rhot1_p .= mat*rhot1*mat'
    for j = 1:Int64(N/2)
        rhot1_p .= matV*rhot1_p*matV'
    end
    rhot1 .= mat'*rhot1_p*mat


    rhot .= rhot1

end

function map_square_expand!(rhot::Matrix{ComplexF64},rhot_p::Matrix{ComplexF64},rhot1::Matrix{ComplexF64},rhot1_p::Matrix{ComplexF64},mat::Matrix{ComplexF64},M::Int64,L::Int64,N::Int64,t::Int64,Nt::Int64)

    rhot_p .= mat*rhot*mat' # rhot in p basis
    rhot1 .= 0.0
    rhot1_p .= 0.0

    for i = 1:L
        for j = 1:L
            i1 = i + M
            j1 = j + M
            i2 = i + M + L
            j2 = j + M + L
            
            # # for n = 0:3  
            # for n = 0:Int64(N/L)-1 ########### check
            #     # rhot1[i1,j1] += rhot[mod1(4*i1+n,N),mod1(4*j1+n,N)]
            #     # rhot1[i2,j2] += rhot[mod1(4*i2+n,N),mod1(4*j2+n,N)]
            #     # rhot1_p[i1,j1] += rhot_p[mod1(4*i1+n,N),mod1(4*j1+n,N)]
            #     # rhot1_p[i2,j2] += rhot_p[mod1(4*i2+n,N),mod1(4*j2+n,N)]
            #     # rhot1[i1,j1] += rhot[mod1(Int64(N/L)*i1+n,N),mod1(Int64(N/L)*j1+n,N)] ########### check
            #     # rhot1[i2,j2] += rhot[mod1(Int64(N/L)*i2+n,N),mod1(Int64(N/L)*j2+n,N)]
            #     # rhot1_p[i1,j1] += rhot_p[mod1(Int64(N/L)*i1+n,N),mod1(Int64(N/L)*j1+n,N)]
            #     # rhot1_p[i2,j2] += rhot_p[mod1(Int64(N/L)*i2+n,N),mod1(Int64(N/L)*j2+n,N)]
            #     rhot1[i1,j1] += rhot[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)] ########### check
            #     rhot1[i2,j2] += rhot[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)]
            #     rhot1_p[i1,j1] += rhot_p[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)]
            #     rhot1_p[i2,j2] += rhot_p[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)]
            # end

            for n = 1:Int64(N/L)
                rhot1[i1,j1] += rhot[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
                rhot1[i2,j2] += rhot[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
                rhot1_p[i1,j1] += rhot_p[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
                rhot1_p[i2,j2] += rhot_p[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n]
            end

        end
    end

    # println(tr(rhot1))
    
    if t == Nt
        rhot .= rhot1*0.25 + mat'*rhot1_p*mat*0.25
    else
        rhot .= 0.0
        rhot_p .= 0.0
        for i = 1:L
            for j = 1:L
                i1 = i + M
                j1 = j + M
                i2 = i + M + L
                j2 = j + M + L

                # # for n = 0:3
                # for n = 0:Int64(N/L)-1 ########### check
                #     # rhot[mod1(4*i1+n,N),mod1(4*j1+n,N)] += rhot1[i1,j1]/4
                #     # rhot[mod1(4*i2+n,N),mod1(4*j2+n,N)] += rhot1[i2,j2]/4
                #     # rhot_p[mod1(4*i1+n,N),mod1(4*j1+n,N)] += rhot1_p[i1,j1]/4
                #     # rhot_p[mod1(4*i2+n,N),mod1(4*j2+n,N)] += rhot1_p[i2,j2]/4
                #     # rhot[mod1(Int64(N/L)*i1+n,N),mod1(Int64(N/L)*j1+n,N)] += rhot1[i1,j1]/Int64(N/L) ########### check
                #     # rhot[mod1(Int64(N/L)*i2+n,N),mod1(Int64(N/L)*j2+n,N)] += rhot1[i2,j2]/Int64(N/L)
                #     # rhot_p[mod1(Int64(N/L)*i1+n,N),mod1(Int64(N/L)*j1+n,N)] += rhot1_p[i1,j1]/Int64(N/L)
                #     # rhot_p[mod1(Int64(N/L)*i2+n,N),mod1(Int64(N/L)*j2+n,N)] += rhot1_p[i2,j2]/Int64(N/L)
                #     rhot[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)] += rhot1[i1,j1]/Int64(N/L) ########### check
                #     rhot[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)] += rhot1[i2,j2]/Int64(N/L)
                #     rhot_p[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)] += rhot1_p[i1,j1]/Int64(N/L)
                #     rhot_p[mod1(Int64(N/L)*i+n,N),mod1(Int64(N/L)*j+n,N)] += rhot1_p[i2,j2]/Int64(N/L)
                # end
                
                for n = 1:Int64(N/L)
                    rhot[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n] += rhot1[i1,j1]/Int64(N/L)
                    rhot[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n] += rhot1[i2,j2]/Int64(N/L)
                    rhot_p[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n] += rhot1_p[i1,j1]/Int64(N/L)
                    rhot_p[Int64(N/L)*(i-1)+n,Int64(N/L)*(j-1)+n] += rhot1_p[i2,j2]/Int64(N/L)
                end

            end
        end
        # println(tr(rhot))
        rhot .= rhot*0.25 + mat'*rhot_p*mat*0.25
    end

end

function map_square_decoupled!(rhot::Matrix{ComplexF64},rhot_p::Matrix{ComplexF64},rhot1::Matrix{ComplexF64},rhot1_p::Matrix{ComplexF64},mat::Matrix{ComplexF64},M::Int64,L::Int64,N::Int64)

    rhot_p .= mat*rhot*mat' # rhot in p basis
    rhot1 .= 0.0
    rhot1_p .= 0.0

    for i = 1:L
        for j = 1:L
            i1 = i + M
            j1 = j + M
            i2 = i + M + L
            j2 = j + M + L
            for n = 0:3  ########### check
                rhot1[i1,j1] += rhot[mod1(4*i1+n,N),mod1(4*j1+n,N)] ########### check
                rhot1[i2,j2] += rhot[mod1(4*i2+n,N),mod1(4*j2+n,N)]
                rhot1_p[i1,j1] += rhot_p[mod1(4*i1+n,N),mod1(4*j1+n,N)]
                rhot1_p[i2,j2] += rhot_p[mod1(4*i2+n,N),mod1(4*j2+n,N)]
            end            
        end
    end

    rhot .= rhot1*0.25 + mat'*rhot1_p*mat*0.25

end

function example_square_decoupled()
    
    # M = 20
    # N = 60 # system size
    # L = 10

    N = 100
    M = Int64(N/4)
    L = Int64(N/4)

    Nt = 1 # number of iteration

    # define matrix to transform q basis to p basis (complex conjugate transposed matrix of this transforms p basis to q basis)
    mat = transform_q2p(N)

    # define initail state rho0
    psi0 = coherent_state(N,0.5,0.5) # can be any state in principle but we choose a cherent state
    rho0 = psi0*psi0'
    rho0 = Matrix{ComplexF64}(I, N, N)/N

    # rho0 = Matrix(I*1.0+1im*0.0, N,N)

    # define density matrices
    rhot = copy(rho0) # time-evolved state to track
    rhot_p = copy(rhot)
    rhot1 = copy(rhot)
    rhot1_p = copy(rhot)
    
    # measure time
    t0 = time()
    println(tr(rhot))

    # iteration
    for t = 1:Nt
        # channel
        # map_square_decoupled!(rhot,rhot_p,rhot1,rhot1_p,mat,M,L,N)
        map_square_expand!(rhot,rhot_p,rhot1,rhot1_p,mat,M,L,N,t,Nt)
        # map_division!(rhot,rhot_p,rhot1,rhot1_p,mat,M,Int64(N/2),N,t,Nt)
        println(tr(rhot))
    end
    
    # plot husimi

    println("preparing for husimi")
    # vec_q = collect(range(0.0, 5.0, length=N))
    # vec_p = copy(vec_q)
    # overlap_qp_mat = husimi_format_2(vec_q,vec_p,N)
    
    # overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    overlap_qp_mat = load("overlap_qp_mat_N100.jld2", "overlap_qp_mat") # run save_overlap_qp_mat first
    vec_q = [1/N:1/N:1;] #[0:1/N:1;] # technically this does not have to be this (I think)
    vec_p = copy(vec_q)
    
    figure()
    f_husimi = husimi(N,rhot,overlap_qp_mat,vec_q,vec_p)
    pcolor(vec_q, vec_p, abs.(f_husimi)')
    colorbar()
    
    println("total time:",round(time()-t0; digits = 3),"sec")

end

function example_square_decoupled_old()
    
    # vecA = [1/4,1/4]
    # vecB = [3/4,1/4]
    # vecC = [1/4,3/4]
    # vecD = [3/4,3/4]

    # M = 10
    # N = 4*M # system size
    # L = copy(M)

    M = 20
    N = 60 # system size
    L = 10

    Nt = 0 #40 # number of iteration

    # define matrix to transform q basis to p basis (complex conjugate transposed matrix of this transforms p basis to q basis)
    mat = transform_q2p(N)

    # define initail state rho0
    psi0 = coherent_state(N,0.2,0.2) # can be any state in principle but we choose a cherent state
    # psi0 .= 0.0
    # psi0[3*M]=1
    # psi0 .= mat'*psi0
    rho0 = psi0*psi0'

    # define density matrices
    rhot = copy(rho0) # time-evolved state to track
    rhot_p = copy(rhot)
    rhot1 = copy(rhot)
    rhot1_p = copy(rhot)
    
    # measure time
    t0 = time()

    # iteration
    for t = 1:Nt
        # channel
        # map_square_decoupled!(rhot,rhot_p,rhot1,rhot1_p,mat,M,L,N)
        map_square_expand!(rhot,rhot_p,rhot1,rhot1_p,mat,M,L,N,t,Nt)
        println(tr(rhot))
    end

    # shearing
    beta = 0.2 # shearing parameter
    map_shearing = zeros(ComplexF64,N,N)
    for n = 1:N
        map_shearing[n,n] = exp(1im*2*pi/N*beta*n*n) # relocate this with amonut the non-integer causes
    end
    map_shearing_p = copy(map_shearing)
    map_shearing_p .= mat'*map_shearing_p*mat

    rhot .= map_shearing*rhot*map_shearing'
    rhot .= map_shearing_p*rhot*map_shearing_p'
    
    # plot husimi

    println("preparing for husimi")
    overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    # overlap_qp_mat = load("overlap_qp_mat_N128.jld2", "overlap_qp_mat") # run save_overlap_qp_mat first
    # vec_q = [1/N:1/N:1;] #[0:1/N:1;] # technically this does not have to be this (I think)
    # vec_p = copy(vec_q)
    
    figure()
    f_husimi = husimi(N,rhot,overlap_qp_mat,vec_q,vec_p)
    pcolor(vec_q, vec_p, abs.(f_husimi)')
    # p = heatmap(vec_q, vec_p, abs.(f_husimi)', aspect_ratio = :equal)
    colorbar()
    # savefig(p,"plot0.png")

    println("total time:",round(time()-t0; digits = 3),"sec")

end

function map_half!(rho1,rho2,L,N)

    rho2 .= 0.0

    for i = 1:L
        for j = 1:L

            ii = i
            jj = j

            for n = 0:1
                rho2[ii,jj] += rho1[mod1(2*ii+n,N),mod1(2*jj+n,N)]
            end
            
        end
    end

end

function test_half()

    L = 2^6
    N = 2*L

    mat = transform_q2p(N)

    # define initail state rho0
    psi0 = coherent_state(N,0.6,0.6) # can be any state in principle but we choose a cherent state
    # psi0 .= 0.0
    # psi0[L]=1
    # psi0 .= mat'*psi0
    rho0 = psi0*psi0'

    rho1 = copy(rho0)
    rho2 = copy(rho0)

    rho1 .= mat*rho1*mat'
    map_half!(rho1,rho2,L,N)
    rho2 .= mat'*rho2*mat
    rho1 .= rho2
    map_half!(rho1,rho2,L,N)
    # rho2 .= mat'*rho2*mat

    # overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    overlap_qp_mat = load("overlap_qp_mat_N128.jld2", "overlap_qp_mat") # run save_overlap_qp_mat first
    vec_q = [0:1/N:1;] # technically this does not have to be this (I think)
    vec_p = copy(vec_q)
    
    figure()
    f_husimi = husimi(N,rho2,overlap_qp_mat)
    pcolor(vec_q, vec_p, abs.(f_husimi)')
    colorbar()

    # return rho2

end

function map_square_test!(rhot::Matrix{ComplexF64},rhot1::Matrix{ComplexF64},rhot1_p::Matrix{ComplexF64},rhot2::Matrix{ComplexF64},rhot2_p::Matrix{ComplexF64},rhot3::Matrix{ComplexF64},rhot3_p::Matrix{ComplexF64},rhot4::Matrix{ComplexF64},rhot4_p::Matrix{ComplexF64},mat::Matrix{ComplexF64},M::Int64,L::Int64,N::Int64)

    # input: rhot

    # update: rhot1, rhot2, rhot3, rhot4 
    #         rhot1_p, rhot2_p, rhot3_p, rhot4_p

    # final output: rhot

    rhot1 .= 0.0
    rhot2 .= 0.0

    for i = 1:L
        for j = 1:L
            i1 = i
            j1 = j
            # i1 = M + i
            # j1 = M + j
            # i2 = 3*M + i
            # j2 = 3*M + j
            for n = 0:1
                rhot1[i1,j1] += rhot[mod1(2*i1+n,N),mod1(2*j1+n,N)] # A, C
                # rhot2[i2,j2] += rhot[mod1(2*i2+n,N),mod1(2*j2+n,N)] # B, D
            end            
        end
    end

    return rhot1

    rhot1 .= mat*rhot1*mat' # rhot1 in momentum basis
    rhot2 .= mat*rhot2*mat'
    rhot3 .= rhot1
    rhot4 .= rhot2

    rhot1_p .= 0.0
    rhot2_p .= 0.0
    rhot3_p .= 0.0
    rhot4_p .= 0.0

    for i = 1:L
        for j = 1:L
            i1 = M + i
            j1 = M + j
            i2 = 3*M + i
            j2 = 3*M + j
            for n = 0:1
                rhot1_p[i1,j1] += rhot1[mod1(2*i1+n,N),mod1(2*j1+n,N)] # A
                rhot2_p[i1,j1] += rhot2[mod1(2*i1+n,N),mod1(2*j1+n,N)] # B
                rhot3_p[i2,j2] += rhot3[mod1(2*i2+n,N),mod1(2*j2+n,N)] # C
                rhot4_p[i2,j2] += rhot4[mod1(2*i2+n,N),mod1(2*j2+n,N)] # D
            end
        end
    end

    rhot1 .= mat'*rhot1_p*mat # rhot1 in position basis
    rhot2 .= mat'*rhot2_p*mat
    rhot3 .= mat'*rhot3_p*mat
    rhot4 .= mat'*rhot4_p*mat

    rhot .= rhot1
    # rhot .= rhot1*0.25 + rhot2*0.25 + rhot3*0.25 + rhot4*0.25

end

function map_square!(rhot::Matrix{ComplexF64},rhot1::Matrix{ComplexF64},rhot1_p::Matrix{ComplexF64},rhot2::Matrix{ComplexF64},rhot2_p::Matrix{ComplexF64},rhot3::Matrix{ComplexF64},rhot3_p::Matrix{ComplexF64},rhot4::Matrix{ComplexF64},rhot4_p::Matrix{ComplexF64},mat::Matrix{ComplexF64},M::Int64,L::Int64,N::Int64)

    # input: rhot

    # update: rhot1, rhot2, rhot3, rhot4 
    #         rhot1_p, rhot2_p, rhot3_p, rhot4_p

    # final output: rhot

    rhot1 .= 0.0
    rhot2 .= 0.0

    for i = 1:L
        for j = 1:L
            i1 = M + i
            j1 = M + j
            i2 = 3*M + i
            j2 = 3*M + j
            for n = 0:1
                rhot1[i1,j1] += rhot[mod1(2*i1+n,N),mod1(2*j1+n,N)] # A, C
                rhot2[i2,j2] += rhot[mod1(2*i2+n,N),mod1(2*j2+n,N)] # B, D
            end            
        end
    end

    rhot1 .= mat*rhot1*mat' # rhot1 in momentum basis
    rhot2 .= mat*rhot2*mat'
    rhot3 .= rhot1
    rhot4 .= rhot2

    rhot1_p .= 0.0
    rhot2_p .= 0.0
    rhot3_p .= 0.0
    rhot4_p .= 0.0

    for i = 1:L
        for j = 1:L
            i1 = M + i
            j1 = M + j
            i2 = 3*M + i
            j2 = 3*M + j
            for n = 0:1
                rhot1_p[i1,j1] += rhot1[mod1(2*i1+n,N),mod1(2*j1+n,N)] # A
                rhot2_p[i1,j1] += rhot2[mod1(2*i1+n,N),mod1(2*j1+n,N)] # B
                rhot3_p[i2,j2] += rhot3[mod1(2*i2+n,N),mod1(2*j2+n,N)] # C
                rhot4_p[i2,j2] += rhot4[mod1(2*i2+n,N),mod1(2*j2+n,N)] # D
            end
        end
    end

    rhot1 .= mat'*rhot1_p*mat # rhot1 in position basis
    rhot2 .= mat'*rhot2_p*mat
    rhot3 .= mat'*rhot3_p*mat
    rhot4 .= mat'*rhot4_p*mat

    rhot .= rhot1
    # rhot .= rhot1*0.25 + rhot2*0.25 + rhot3*0.25 + rhot4*0.25

end

function example_square()
    
    M = 2^4
    L = 4*M
    N = 2*L # system size
    Nt = 1 # number of iteration

    # define matrix to transform q basis to p basis (complex conjugate transposed matrix of this transforms p basis to q basis)
    mat = transform_q2p(N)

    # define initail state rho0
    psi0 = coherent_state(N,0.5,0.5) # can be any state in principle but we choose a cherent state
    psi0 .= 0.0
    psi0[3*M]=1
    # psi0 .= mat'*psi0
    rho0 = psi0*psi0'

    # define density matrices
    rhot = copy(rho0) # time-evolved state to track
    rhot1 = copy(rhot)
    rhot1_p = copy(rhot)
    rhot2 = copy(rhot)
    rhot2_p = copy(rhot)
    rhot3 = copy(rhot)
    rhot3_p = copy(rhot)
    rhot4 = copy(rhot)
    rhot4_p = copy(rhot)

    # measure time
    t0 = time()

    # iteration
    for t = 1:Nt
        # channel
        map_square_test!(rhot,rhot1,rhot1_p,rhot2,rhot2_p,rhot3,rhot3_p,rhot4,rhot4_p,mat,M,L,N)
    end
    
    # plot husimi

    println("preparing for husimi")
    # overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    overlap_qp_mat = load("overlap_qp_mat_N128.jld2", "overlap_qp_mat") # run save_overlap_qp_mat first
    vec_q = [0:1/N:1;] # technically this does not have to be this (I think)
    vec_p = copy(vec_q)
    
    figure()
    f_husimi = husimi(N,rhot,overlap_qp_mat)
    pcolor(vec_q, vec_p, abs.(f_husimi)')
    colorbar()

    println("total time:",round(time()-t0; digits = 3),"sec")

end

#=-----------------------------------------------------------------------------#
BAKER MAP
#-----------------------------------------------------------------------------=#

function example_baker_2()

    L = 50
    N = 2*L # system size
    M1 = 20
    M2 = N - M1

    Nt = 100 # number of iteration

    # define matrix to transform q basis to p basis (complex conjugate transposed matrix of this transforms p basis to q basis)
    matN = transform_q2p(N)
    matM1 = transform_q2p(M1)
    matM2 = transform_q2p(M2)

    # define initail state rho0
    psi0 = coherent_state(N,0.5,0.5) # can be any state in principle but we choose a cherent state
    # psi0 .= 0.0
    # psi0[L]=1
    # psi0 .= matN'*psi0
    rho0 = psi0*psi0'

    # define density matrices
    rhot = copy(rho0) # time-evolved state to track
    
    # measure time
    t0 = time()

    # iteration
    matM1_expand = zeros(ComplexF64,N,N)
    matM1_expand[1:M1,1:M1] = matM1
    matM2_expand = zeros(ComplexF64,N,N)
    matM2_expand[M1+1:M1+M2,M1+1:M1+M2] = matM2
    map_baker = matN' * (matM1_expand + matM2_expand)
    # map_splitt = kron(mat_I,matN2)
    # map_baker = matN' * map_splitt
    for t = 1:Nt
        rhot .= map_baker*rhot*map_baker'
    end
    
    # plot husimi

    println("preparing for husimi")
    # overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    overlap_qp_mat = load("overlap_qp_mat_N100.jld2", "overlap_qp_mat") # run save_overlap_qp_mat first
    vec_q = [1/N:1/N:1;] #[0:1/N:1;] # technically this does not have to be this (I think)
    vec_p = copy(vec_q)
    
    figure()
    f_husimi = husimi(N,rhot,overlap_qp_mat,vec_q,vec_p)
    pcolor(vec_q, vec_p, abs.(f_husimi)')
    colorbar()

    println("total time:",round(time()-t0; digits = 3),"sec")

end

function example_baker_1()

    L = 50
    N = 2*L # system size
    Nt = 100 # number of iteration

    # define matrix to transform q basis to p basis (complex conjugate transposed matrix of this transforms p basis to q basis)
    matN = transform_q2p(N)
    matN2 = transform_q2p(Int64(N/2))

    # define initail state rho0
    psi0 = coherent_state(N,0.5,0.5) # can be any state in principle but we choose a cherent state
    # psi0 .= 0.0
    # psi0[20]=1
    # psi0 .= matN'*psi0
    rho0 = psi0*psi0'

    # rho0 = Matrix{ComplexF64}(I, N, N)/N

    # define density matrices
    rhot = copy(rho0) # time-evolved state to track
    
    # measure time
    t0 = time()

    # iteration
    mat_I = Matrix(I*1, 2,2)
    map_splitt = kron(mat_I,matN2)
    map_baker = matN' * map_splitt
    for t = 1:Nt
        rhot .= map_baker*rhot*map_baker'
    end

    println(tr(rho0))
    
    # plot husimi

    println("preparing for husimi")
    # overlap_qp_mat, vec_q, vec_p = husimi_format(N)
    overlap_qp_mat = load("overlap_qp_mat_N100.jld2", "overlap_qp_mat") # run save_overlap_qp_mat first
    vec_q = [1/N:1/N:1;] #[0:1/N:1;] # technically this does not have to be this (I think)
    vec_p = copy(vec_q)
    
    figure()
    f_husimi = husimi(N,rhot,overlap_qp_mat,vec_q,vec_p)
    pcolor(vec_q, vec_p, abs.(f_husimi)')
    colorbar()

    println("total time:",round(time()-t0; digits = 3),"sec")

end

################### no need to see ###################

function map_G!(rhot,rhot1,L::Int64,N::Int64,ind::Int64)
    # (rhot::Matrix{ComplexF64},rhot1::Matrix{ComplexF64},L::Int64,N::Int64,ind::Int64)

    rhot1 .= 0.0

    for i = 1:L
        for j = 1:L

            # ii = 2*L*mod(ind+1,2)+i
            # jj = 2*L*mod(ind+1,2)+j
            ii = i
            jj = j

            for n = 0:2
                rhot1[ii,jj] += rhot[mod1(3*ii+n,N),mod1(3*jj+n,N)]
            end
            
        end
    end

end

function fidelity(rho,sigma)

    return tr(sqrt(sqrt(rho)*sigma*sqrt(rho)))^2

end

function test_commute()

    L = 3^3
    N = 3*L
    
    psi0 = coherent_state(N,0.5,0.5)
    # psi0 .= 0.0
    # psi0[L] = 1
    rho0 = psi0*psi0'

    rhot = copy(rho0)
    # rhot_p = copy(rho0)
    rhot1 = copy(rhot)
    # rhot1_p = copy(rhot)
    
    # define matrix to transform q basis to p basis (complex conjugate transposed matrix of this transforms p basis to q basis)
    mat = transform_q2p(N)

    # map_G!(rhot,rhot1,L,N,1) # position basis
    # rhot .= rhot1
    # rhot .= mat*rhot*mat'
    # map_G!(rhot,rhot1,L,N,1) # momentum basis
    # rhot .= rhot1
    # rhot .= mat'*rhot*mat # position basis

    rhot .= mat*rhot*mat'
    map_G!(rhot,rhot1,L,N,1) # momentum basis
    rhot .= rhot1
    rhot .= mat'*rhot*mat
    map_G!(rhot,rhot1,L,N,1) # position basis
    rhot .= rhot1
    
    # rhot .= copy(rho0)
    # map_G!(rhot,rhot1,L,N,1) # position basis
    # rhot .= rhot1 
    # rhot .= mat*rhot*mat'
    # map_G!(rhot,rhot1,L,N,1) # momentum basis
    # rhot .= rhot1 
    # rho1 = mat'*rhot*mat

    # return rhotA, rhotB, mat
    # return fidelity(rhotA,rhotB)

    # # plot husimi
    overlap_qp_mat = load("overlap_qp_mat_L27.jld2", "overlap_qp_mat") # run save_overlap_qp_mat first
    vec_q = [0:1/N:1;] # technically this does not have to be this (I think)
    vec_p = copy(vec_q)
    
    f_husimi = husimi(N,rhot,overlap_qp_mat)
    # f_husimiA = husimi(N,rhotA,overlap_qp_mat)
    # f_husimiB = husimi(N,rhotB,overlap_qp_mat)
    # f_husimi_1 = husimi(N,rho1,overlap_qp_mat)
    
    figure()
    pcolor(vec_q, vec_p, abs.(f_husimi)')
    # pcolor(vec_q, vec_p, abs.(f_husimiA)')
    # pcolor(vec_q, vec_p, abs.(f_husimiB)')
    # pcolor(vec_q, vec_p, abs.(f_husimi_1)')
    colorbar()

    # return sum(f_husimiA)/N

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