import matplotlib.pyplot as plt
import qiskit
import numpy
import random
import time
import gc
from scipy.linalg import expm, block_diag
from scipy.linalg import sqrtm
from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import DensityMatrix, state_fidelity, Kraus, Stinespring, Operator, Statevector
from qiskit_experiments.library import StateTomography

#------------------------------------------------------------------------------#
# AUX
#  Notes: Regenerating matA's... meh
#------------------------------------------------------------------------------#

def make_filename(i):
    return "dens%04d.npy" % i

def make_fid_filename(i):
    return "fids%04d.csv" % i

def rotate_in_phase_space(val, theta):
    re = val.real*numpy.cos(theta) - val.imag*numpy.sin(theta)
    im = val.real*numpy.sin(theta) + val.imag*numpy.cos(theta)
    return re + im*1j

def generate_mat_displacement(N,alpha):
    matAdag = numpy.zeros((N, N), dtype=complex)
    for i in range(N-1):
        matAdag[i+1, i] = numpy.sqrt(i+1)
    matA = matAdag.T

    matD = expm(alpha/numpy.sqrt(2) * matAdag - numpy.conj(alpha)/numpy.sqrt(2) * matA)
    return matD

def generate_mat_rotation(N, theta):
    matN = numpy.diag(range(N))
    matR = expm(theta*matN*1j)
    return matR

def generate_mat_squeeze(N, r, theta):
    matAdag = numpy.zeros((N, N), dtype=complex)
    for i in range(N-1):
        matAdag[i+1, i] = numpy.sqrt(i+1)
    matA = matAdag.T

    chi = r * numpy.exp(theta * 1j)

    matS = (numpy.linalg.matrix_power(numpy.conj(chi)*matA,2) - numpy.linalg.matrix_power(chi*matAdag,2))/2
    matS = expm(matS)

    return matS

def generate_mat_shearingx2(N, beta):
    matAdag = numpy.zeros((N, N), dtype=complex)
    for i in range(N-1):
        matAdag[i+1, i] = numpy.sqrt(i+1)
    matA = matAdag.T

    matX = (matAdag+matA)/numpy.sqrt(2)
    matP = 1j*(matAdag-matA)/numpy.sqrt(2)

    matS = expm(1j*(beta/3)*numpy.linalg.matrix_power(matX,3))

    return matS

def generate_mat_shearingx3(N, beta):
    matAdag = numpy.zeros((N, N), dtype=complex)
    for i in range(N-1):
        matAdag[i+1, i] = numpy.sqrt(i+1)
    matA = matAdag.T

    matX = (matAdag+matA)/numpy.sqrt(2)
    matP = 1j*(matAdag-matA)/numpy.sqrt(2)
    
    matS = expm(1j*(beta/3)*numpy.linalg.matrix_power(matP,3))
        
    return matS


def generate_mat_d_rotation(nqpm, theta1, theta2):
    N = 2**nqpm
    matD1 = generate_mat_rotation(N, theta1)
    matD2 = generate_mat_rotation(N, theta2)
    return block_diag(matD1,matD2)

def generate_mat_d_squeeze(nqpm, r, theta):
    N = 2**nqpm
    matD1 = generate_mat_squeeze(N, r, theta)
    return block_diag(matD1,matD1)

def generate_mat_d_shearingx2(nqpm, beta):
    N = 2**nqpm
    matD1 = generate_mat_shearingx2(N, beta)
    return block_diag(matD1,matD1)

def generate_mat_d_displacement(nqpm,alpha1,alpha2):
    N = 2**nqpm
    matD1 = generate_mat_displacement(N,alpha1)
    matD2 = generate_mat_displacement(N,alpha2)
    return block_diag(matD1,matD2)

def make_displacement_circuit(x, y, xmax, ymax, num_bits, max_value = 2*numpy.pi):
    x_ratio = max_value*x/xmax
    y_ratio = max_value*y/xmax
    alpha = -0.5*max_value+x_ratio+(-0.5*max_value+y_ratio)*1j

    d = 2**num_bits

    mat_displacement = generate_mat_displacement(d, alpha)

    gate_displacement = UnitaryGate(mat_displacement, label="displacement_operator")
    qc = QuantumCircuit(num_bits)
    qc.append(gate_displacement, [i for i in range(num_bits)])
    return qc

#------------------------------------------------------------------------------#
# HUSIMI
#------------------------------------------------------------------------------#

# maybe partial trace?
def simple_husimi(xmax, ymax, dens_mat, nqpm, max_value = 2*numpy.pi):
    fids = numpy.zeros((xmax, ymax))

    c_ref = init_gaussian(nqpm)

    for i in range(0,xmax):
        for j in range(0,ymax):
            c_move_ref = make_displacement_circuit(i, j, xmax, ymax, nqpm, max_value = max_value)
            state_ref = Statevector(c_move_ref)
            dens_mat_ref = qiskit.quantum_info.DensityMatrix(state_ref)

            fids[i][j] = state_fidelity(dens_mat, dens_mat_ref)
    return fids

def find_fidelity(mat1, mat2):
    sqrt_mat1 = sqrtm(mat1)
    #mid_mat = numpy.trace(sqrtm(sqrt_mat1 @ mat2 @ sqrt_mat1))
    #mid_mat = numpy.matmul(mid_mat, mid_mat)
    #return numpy.real(mid_mat)
    return numpy.real(numpy.linalg.matrix_power(numpy.trace(sqrtm(sqrt_mat1 @ mat2 @ sqrt_mat1)),2))

def husimi_with_duplicates(xmax, ymax, dens_mat, nqpm, max_value = 2*numpy.pi, num_ancillary = 3):
    total_objects = 2**num_ancillary
    fids = numpy.zeros((xmax, ymax))

    c_ref = init_gaussian(nqpm)

    for i in range(0,xmax):
        for j in range(0,ymax):
            c_move_ref = make_displacement_circuit(i, j, xmax, ymax, nqpm, max_value = max_value)
            state_ref = Statevector(c_move_ref)
            dens_mat_ref = qiskit.quantum_info.DensityMatrix(state_ref)

            for k in range(total_objects):
                stride = 2**nqpm
                dens_mat_mini = dens_mat.data[k*stride:(k+1)*stride, k*stride:(k+1)*stride]
                new_fid = find_fidelity(dens_mat_mini, dens_mat_ref.data) / total_objects

                fids[i][j] = fids[i][j] + new_fid
    return fids

#------------------------------------------------------------------------------#
# HARDWARE
#------------------------------------------------------------------------------#

def find_backend():
    # Noisy simulator
    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")
    return AerSimulator.from_backend(backend)

#------------------------------------------------------------------------------#
# INIT
#------------------------------------------------------------------------------#

def init_gaussian(nqpm, ancillary_bits = 0):
    return QuantumCircuit(nqpm + ancillary_bits)

# swirling
def init_circle_set(N, M, num_circuits):
    thetas = numpy.random.uniform(0, numpy.pi, num_circuits)
    #thetas = numpy.linspace(0, 2*numpy.pi, num_circuits)

    circuits = []
    for theta in thetas:
        qc = QuantumCircuit(N)
        for n in range(M):
            qc.h(n)
            choice = random.choice([0,1,2,3])
            if choice == 1:
                qc.rx(theta*2, n)
            elif choice == 2:
                qc.ry(theta*2, n)
            elif choice == 3:
                qc.rz(theta, n)
            qc.h(n)
        circuits.append(qc)
    return circuits

# N Number of qubits
# M size of circle M < N
def init_circle_aer(N,M):
    dephasing_ops = [
        numpy.array([[1, 0], [0, 0]]),
        numpy.array([[0, 0], [0, 1]])
    ]
    dephasing_channel = Kraus(dephasing_ops)

    qc = QuantumCircuit(N)

    for n in range(M):
        qc.h(n)
        qc.append(dephasing_channel.to_instruction(), [n])

    return qc

def init_ring(N):
    qc = QuantumCircuit(N)
    qc.x(N-1)

    return qc

#------------------------------------------------------------------------------#
# ANIMATION
#------------------------------------------------------------------------------#
def shear_electrons(nqpm, beta):
    N = 2**nqpm

    matD = generate_mat_shearingx2(N, beta)
    matD2 = numpy.eye(2**nqpm)

    return block_diag(matD,matD, matD, matD, matD, matD, matD2, matD2)

def displace_electrons(nqpm, d):
    N = 2**nqpm

    matD = generate_mat_displacement(N, d)
    matD2 = numpy.eye(2**nqpm)
    return block_diag(matD,matD, matD, matD, matD, matD, matD2, matD2)

def rotate_electrons(nqpm, time_ratio):
    N = 2**nqpm

    offset = numpy.pi*time_ratio

    matD1 = generate_mat_rotation(N, offset)
    matD2 = generate_mat_rotation(N,numpy.pi/3+offset)

    matD3 = generate_mat_rotation(N,2*numpy.pi/3+offset)
    matD4 = generate_mat_rotation(N,numpy.pi+offset)

    matD5 = generate_mat_rotation(N,4*numpy.pi/3+offset)
    matD6 = generate_mat_rotation(N,5*numpy.pi/3+offset)

    matD7 = matD1
    matD8 = matD1
    return block_diag(matD1,matD2, matD3, matD4, matD5, matD6, matD7, matD8)

def squeeze_electrons(nqpm, scale, theta = 0):
    N = 2**nqpm

    dist = numpy.log(scale)
    # top and bottom
    matD1 = generate_mat_squeeze(N, dist, 0)
    matD4 = generate_mat_squeeze(N, dist, 0)

    # one arc
    #matD2 = generate_mat_squeeze(N, dist, numpy.pi/3)
    #matD5 = generate_mat_squeeze(N, dist, numpy.pi/3)
    matD2 = generate_mat_squeeze(N, dist, 0)
    matD5 = generate_mat_squeeze(N, dist, 0)

    # the other arc
    #matD3 = generate_mat_squeeze(N, dist, 2*numpy.pi/3)
    #matD6 = generate_mat_squeeze(N, dist, 2*numpy.pi/3)
    matD3 = generate_mat_squeeze(N, dist, 0)
    matD6 = generate_mat_squeeze(N, dist, 0)

    # both nucleii
    matD7 = numpy.eye(2**nqpm)
    matD8 = numpy.eye(2**nqpm)
    return block_diag(matD1,matD2, matD3, matD4, matD5, matD6, matD7, matD8)

def rotate_electrons_2(nqpm):

    N = 2**nqpm

    # top and bottom
    matD1 = numpy.eye(2**nqpm)
    matD4 = matD1

    # one arc
    matD2 = generate_mat_rotation(N, numpy.pi/3)
    matD5 = generate_mat_rotation(N, numpy.pi/3)

    # the other arc
    matD3 = generate_mat_rotation(N, 2*numpy.pi/3)
    matD6 = generate_mat_rotation(N, 2*numpy.pi/3)

    # both nucleii
    matD7 = matD1
    matD8 = matD1
    return block_diag(matD1,matD2, matD3, matD4, matD5, matD6, matD7, matD8)


def generate_electrons(nqpm, radius):

    N = 2**nqpm

    # top and bottom
    #matD1 = numpy.eye(2**nqpm)
    #matD2 = numpy.eye(2**nqpm)
    matD1 = generate_mat_displacement(N, radius*1j)
    matD2 = generate_mat_displacement(N,-radius*1j)

    # one arc
    #matD3 = numpy.eye(2**nqpm)
    #matD4 = numpy.eye(2**nqpm)
    matD3 = generate_mat_displacement(N, rotate_in_phase_space(radius*1j, numpy.pi/3))
    matD4 = generate_mat_displacement(N, rotate_in_phase_space(radius*1j, -2*numpy.pi/3))

    # the other arc
    #matD5 = numpy.eye(2**nqpm)
    #matD6 = numpy.eye(2**nqpm)
    matD5 = generate_mat_displacement(N, rotate_in_phase_space(radius*1j, 2*numpy.pi/3))
    matD6 = generate_mat_displacement(N, rotate_in_phase_space(radius*1j, -numpy.pi/3))

    # both nucleii
    matD7 = numpy.eye(2**nqpm)
    matD8 = numpy.eye(2**nqpm)
    return block_diag(matD1,matD2, matD3, matD4, matD5, matD6, matD7, matD8)

#------------------------------------------------------------------------------#
# QIFS ROUTINES
#------------------------------------------------------------------------------#

def qifs_check(num_shots = 1024, nqpm = 4, max_value = 4*numpy.pi):
    start = time.time()
    d0 = 2**nqpm

    ancillary_bits = 1

    qubit_list = list(range(nqpm + ancillary_bits))
    qubit_of_interest_list = list(range(nqpm))

    init_state_c = init_gaussian(nqpm, ancillary_bits = ancillary_bits)
    #init_state_c = init_circle_aer(nqpm+ancillary_bits, nqpm-2)
    init_state_c.h(nqpm+ancillary_bits-1)
    backend = AerSimulator()

    mat_shearingx2 = generate_mat_d_shearingx2(nqpm, 0.333)
    gate_shearingx2 = UnitaryGate(mat_shearingx2, label = "shearingx2")
    init_state_c.append(gate_shearingx2, qubit_list)

    mat_displacement = generate_mat_d_displacement(nqpm, 3.0j, 3.0)
    gate_displacement = UnitaryGate(mat_displacement, label="displacement")
    init_state_c.append(gate_displacement, qubit_list)

    mat_rot = generate_mat_d_rotation(nqpm, 0.5, -.5)
    gate_rot = UnitaryGate(mat_rot, label="rot")
    init_state_c.append(gate_rot, qubit_list)

    mat_squeeze = generate_mat_d_squeeze(nqpm, numpy.log(2.0), 0.5)
    mat_squeeze = generate_mat_d_squeeze(nqpm, numpy.log(10.0), 0.0)
    gate_squeeze = UnitaryGate(mat_squeeze, label = "squeeze")
    #init_state_c.append(gate_squeeze, qubit_list)

    #tomo_state = StateTomography(init_state_c, measurement_indices=qubit_list)
    tomo_state = StateTomography(init_state_c, measurement_indices=qubit_of_interest_list)

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    print(time.time() - start)
    print("Finding husimi...")
    #fids = husimi_with_duplicates(50,50, dens_mat_state, nqpm, max_value = max_value, num_ancillary = 1)
    fids = simple_husimi(50,50, dens_mat_state, nqpm, max_value = max_value)

    #return (fids, dens_mat_state)

    print(time.time() - start)
    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    return fids

def qifs_animation(num_shots = 1024, nqpm = 4, total_time = 1, frame = 12):

    # max_value = 2**nqpm
    if nqpm == 4:
        displacement = -4.5j
        max_value = 10
    else:
        displacement = -5.0j
        max_value = 14

    start = time.time()
    d0 = 2**nqpm

    ancillary_bits = 3

    qubit_list = list(range(nqpm + ancillary_bits))
    qubit_of_interest_list = list(range(nqpm))

    print("curr frame: %d" %frame)

    time_ratio = frame / 30
    init_state_c = init_gaussian(nqpm, ancillary_bits = ancillary_bits)
    #init_state_c = init_circle_aer(nqpm+ancillary_bits, 2)
    for i in range(ancillary_bits):
        init_state_c.h(nqpm+ancillary_bits-1-i)

    backend = AerSimulator()

    matAdag = numpy.zeros((d0, d0), dtype=complex)
    for i in range(d0-1):
        matAdag[i+1, i] = numpy.sqrt(i+1)
    matA = matAdag.T

    # shear
    #shear_mat = shear_electrons(nqpm, 0.3333333)
    shear_mat = shear_electrons(nqpm, 2)
    shear_gate = UnitaryGate(shear_mat, label="shearing")
    init_state_c.append(shear_gate, qubit_list)

    # displace
    displacement_mat = displace_electrons(nqpm, displacement)
    displacement_gate = UnitaryGate(displacement_mat, label="displace")
    init_state_c.append(displacement_gate, qubit_list)

    # rotate
    rotation_mat = rotate_electrons(nqpm, time_ratio)
    rotation_gate = UnitaryGate(rotation_mat, label="rotate")
    init_state_c.append(rotation_gate, qubit_list)

    # squeeze
    squeeze_mat = squeeze_electrons(nqpm, 1.5)
    squeeze_gate = UnitaryGate(squeeze_mat, label="squeeze")
    init_state_c.append(squeeze_gate, qubit_list)

    # rotate
    rotation_2_mat = rotate_electrons_2(nqpm)
    rotation_2_gate = UnitaryGate(rotation_2_mat, label="rotate_2")
    init_state_c.append(rotation_2_gate, qubit_list)

    #electrons = generate_electrons(nqpm, 4)
    #gate_electrons = UnitaryGate(electrons, label="electrons")
    #init_state_c.append(gate_electrons, qubit_list)

    print(time.time() - start)
    print("Running tomography...")
    #tomo_state = StateTomography(init_state_c, measurement_indices=qubit_list)
    tomo_state = StateTomography(init_state_c, measurement_indices=qubit_of_interest_list)

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    filename = "density_%d_%04d.npy"%(nqpm, frame)
    numpy.save(filename, dens_mat_state.data)

    #return dens_mat_state

    print(time.time() - start)
    print("Finding husimi...")
    #fids = husimi_with_duplicates(50,50, dens_mat_state, nqpm, max_value = max_value)
    #fids = simple_husimi(128, 128, dens_mat_state, nqpm, max_value = max_value)
    fids = simple_husimi(50,50, dens_mat_state, nqpm, max_value = max_value)

    print(time.time() - start)
    print("Plotting...")
    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    print("Outputting...")
    fid_filename = "fids_%d_%04d.csv"%(nqpm, frame)
    numpy.savetxt(fid_filename, fids.T, delimiter=",")


def qifs_gaussian(num_shots = 1024, nqpm = 4, max_value = 2*numpy.pi):
    start = time.time()
    init_state_c = init_gaussian(nqpm)
    backend = find_backend()

    print(time.time() - start)
    print("Running tomography...")
    tomo_state = StateTomography(init_state_c, measurement_indices=[i for i in range(0, nqpm)])

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    print(time.time() - start)
    print("Finding husimi...")
    fids = simple_husimi(50,50, dens_mat_state, nqpm, max_value = max_value)
    
    print(time.time() - start)
    print("Plotting...")
    filename = make_filename(0)
    numpy.save(filename, dens_mat_state.data)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    return fids

def qifs_ideal(num_shots = 1024, nqpm = 5):
    init_state_c = init_circle_aer(nqpm, 1)
    backend = AerSimulator()

    tomo_state = StateTomography(init_state_c, measurement_indices=[i for i in range(0, nqpm)])

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    print("Finding husimi...")
    fids = simple_husimi(50,50, dens_mat_state, nqpm, max_value = 16)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    return fids

def generate_mat_smiley_squeeze(nqpm):
    N = 2**nqpm
    matD1 = generate_mat_squeeze(N, numpy.log(1.5)*1j, 0)
    matD2 = generate_mat_squeeze(N, numpy.log(1.5)*1j, 0)
    matD3 = generate_mat_squeeze(N, numpy.log(3), 0)
    return block_diag(matD1, matD2, matD3, matD3)

def generate_mat_smiley_displacement(nqpm,alpha1,alpha2, alpha3):
    N = 2**nqpm
    matD1 = generate_mat_displacement(N,alpha1)
    matD2 = generate_mat_displacement(N,alpha2)
    matD3 = generate_mat_displacement(N,alpha3)
    return block_diag(matD1, matD2, matD3, matD3)

def generate_mat_smiley_shearingx3(nqpm):
    N = 2**nqpm
    matD1 = numpy.eye(N)
    matD2 = generate_mat_shearingx3(N,0.125)
    return block_diag(matD1, matD1, matD2, matD2)

def qifs_smiley(num_shots = 1024, nqpm = 5):
    if nqpm == 5:
        world_size = 14
        state_c = init_circle_aer(nqpm+2, 1)
        scale = 1.3
    elif nqpm == 4:
        world_size = 10 
        state_c = init_gaussian(nqpm, ancillary_bits = 2)
        scale = 1
    else:
        print("number of modes not supported!")

    start = time.time()
    state_c.h(nqpm+1)
    state_c.h(nqpm)

    qubit_list = list(range(nqpm+2))
    qubit_of_interest_list = list(range(nqpm))

    # squeeze
    squeeze_mat = generate_mat_smiley_squeeze(nqpm)
    squeeze_gate = UnitaryGate(squeeze_mat, label="squeeze")
    state_c.append(squeeze_gate, qubit_list)

    # smile
    mat_shearingx3 = generate_mat_smiley_shearingx3(nqpm)
    gate_shearingx3 = UnitaryGate(mat_shearingx3, label = "shearingx3")
    state_c.append(gate_shearingx3, qubit_list)

    # displace
    displacement_mat = generate_mat_smiley_displacement(nqpm, scale*(-2.5-2j),scale*(-2.5+2j),scale * 3)
    displacement_gate = UnitaryGate(displacement_mat, label="displace")
    state_c.append(displacement_gate, qubit_list)

    #backend = find_backend()
    backend = AerSimulator()

    print(time.time() - start)
    print("Running tomography...")
    tomo_state = StateTomography(state_c, measurement_indices=qubit_of_interest_list)
    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()
    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    filename = "smiley_%d.npy" %nqpm
    numpy.save(filename, dens_mat_state.data)

    print(time.time() - start)
    print("Finding husimi...")

    fids = simple_husimi(50,50, dens_mat_state, nqpm, max_value = world_size)

    print(time.time() - start)
    print("Plotting...")
    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    print("Outputting...")
    fid_filename = "smiley_%d.csv"%nqpm
    numpy.savetxt(fid_filename, fids.T, delimiter=",")


def qifs(num_shots=128, nqpm = 5, occupied_states = 2, num_states = 10):
    '''
    (qmr0, init_state_c) = init_gaussian(nqpm)

    init_state_ch = decompose_init_circuit(init_state_c)

    '''
    init_states_c = init_circle_set(nqpm, occupied_states, num_states)
    backend = find_backend()

    tomo_states = [StateTomography(init_states_c[j], measurement_indices=[i for i in range(0, nqpm)]) for j in range(num_states)]

    result_states = [tomo_states[i].run(backend, shots = num_shots).block_for_results() for i in range(num_states)]

    dens_mat_states = [result_states[i].analysis_results("state", dataframe = True).iloc[0].value for i in range(num_states)]
    dens_mat_state = (1/num_states)*dens_mat_states[0]
    for i in range (1,num_states):
        dens_mat_state = dens_mat_state + (1/num_states)*dens_mat_states[i]

    print("Finding husimi...")
    fids = simple_husimi(50,50, dens_mat_state, nqpm)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    #ideal_backend = AerSimulator()
    #tomo_state_ideal = StateTomography(init_state_c, measurement_indices=[0])

    #dens_mat_state_ideal = tomo_state_ideal.run(ideal_backend, shots = num_shots).block_for_results().analysis_results("state", dataframe = True).iloc[0].value
    #fids = simple_husimi(5, 5, dens_mat_state)

    return fids

if __name__ == "__main__":
    #qifs_gaussian(nqpm = 3, num_shots = 1024)
    #qifs_ideal()
    #qifs_check(nqpm = 5)
    #qifs_animation()
    #qifs(num_states = 20)
    #for i in range(30):
    #    qifs_animation(frame = i)
    #    gc.collect()
    qifs_smiley()
