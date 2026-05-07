

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
