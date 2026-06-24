import matplotlib.pyplot as plt
import qiskit
import numpy
import random
import time
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

def rotate_in_phase_space(val, theta):
    re = val.real*numpy.cos(theta) - val.imag*numpy.sin(theta)
    im = val.real*numpy.sin(theta) + val.imag*numpy.cos(theta)
    return re + im*1j

def generate_mat_displacement(nqpm,alpha):
    N = 2**nqpm
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

    matS = (numpy.conj(chi)*matA**2 - chi*matAdag**2)/2
    matS = expm(matS)

    return matS

def generate_mat_shearingx2(N, beta):
    matAdag = numpy.zeros((N, N), dtype=complex)
    for i in range(N-1):
        matAdag[i+1, i] = numpy.sqrt(i+1)
    matA = matAdag.T

    matX = (matAdag+matA)/numpy.sqrt(2)
    matP = 1j*(matAdag-matA)/numpy.sqrt(2)

    matS = expm(1j*(beta/3)*matX**3)

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
    matD1 = generate_mat_displacement(nqpm,alpha1)
    matD2 = generate_mat_displacement(nqpm,alpha2)
    return block_diag(matD1,matD2)

def make_displacement_circuit(x, y, xmax, ymax, num_bits, max_value = 2*numpy.pi):
    x_ratio = max_value*x/xmax
    y_ratio = max_value*y/xmax
    alpha = -0.5*max_value+x_ratio+(-0.5*max_value+y_ratio)*1j

    d = 2**num_bits

    mat_displacement = generate_mat_displacement(num_bits, alpha)

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
    return numpy.real(numpy.trace(sqrtm(sqrt_mat1 @ mat2 @ sqrt_mat1))**2)

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
def generate_electrons(matA, matAdag, radius, nqpm):

    # top and bottom
    #matD1 = numpy.eye(2**nqpm)
    #matD2 = numpy.eye(2**nqpm)
    matD1 = generate_mat_displacement(matA,matAdag, radius*1j)
    matD2 = generate_mat_displacement(matA,matAdag,-radius*1j)

    # one arc
    #matD3 = numpy.eye(2**nqpm)
    #matD4 = numpy.eye(2**nqpm)
    matD3 = generate_mat_displacement(matA, matAdag, rotate_in_phase_space(radius*1j, numpy.pi/3))
    matD4 = generate_mat_displacement(matA,matAdag, rotate_in_phase_space(radius*1j, -2*numpy.pi/3))

    # the other arc
    #matD5 = numpy.eye(2**nqpm)
    #matD6 = numpy.eye(2**nqpm)
    matD5 = generate_mat_displacement(matA,matAdag, rotate_in_phase_space(radius*1j, 2*numpy.pi/3))
    matD6 = generate_mat_displacement(matA,matAdag, rotate_in_phase_space(radius*1j, -numpy.pi/3))

    # both nucleii
    matD7 = numpy.eye(2**nqpm)
    matD8 = numpy.eye(2**nqpm)
    return block_diag(matD1,matD2, matD3, matD4, matD5, matD6, matD7, matD8)

#------------------------------------------------------------------------------#
# QIFS ROUTINES
#------------------------------------------------------------------------------#

def qifs_check(num_shots = 1024, nqpm = 6, max_value = 6*numpy.pi):
    start = time.time()
    d0 = 2**nqpm

    ancillary_bits = 1

    qubit_list = list(range(nqpm + ancillary_bits))

    init_state_c = init_gaussian(nqpm, ancillary_bits = ancillary_bits)
    init_state_c.h(nqpm+ancillary_bits-1)
    backend = AerSimulator()

    mat_shearingx2 = generate_mat_d_shearingx2(nqpm, 10.0)
    gate_shearingx2 = UnitaryGate(mat_shearingx2, label = "shearingx2")
    init_state_c.append(gate_shearingx2, qubit_list)

    mat_displacement = generate_mat_d_displacement(nqpm, 3.0j, 3.0j)
    gate_displacement = UnitaryGate(mat_displacement, label="displacement")
    init_state_c.append(gate_displacement, qubit_list)

    mat_rot = generate_mat_d_rotation(nqpm, 0.5, -.5)
    gate_rot = UnitaryGate(mat_rot, label="rot")
    init_state_c.append(gate_rot, qubit_list)

    #mat_squeeze = generate_mat_d_squeeze(nqpm, 1.0, 0.0)
    #gate_squeeze = UnitaryGate(mat_squeeze, label = "squeeze")
    #init_state_c.append(gate_squeeze, qubit_list)

    tomo_state = StateTomography(init_state_c, measurement_indices=qubit_list)

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    print(time.time() - start)
    print("Finding husimi...")
    fids = husimi_with_duplicates(50,50, dens_mat_state, nqpm, max_value = max_value, num_ancillary = 1)

    #return (fids, dens_mat_state)

    print(time.time() - start)
    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    return fids



def qifs_animation(num_shots = 1024, nqpm = 4, max_value = 4*numpy.pi):

    d0 = 2**nqpm

    ancillary_bits = 3

    qubit_list = list(range(nqpm + ancillary_bits))

    init_state_c = init_gaussian(nqpm, ancillary_bits = ancillary_bits)
    for i in range(ancillary_bits):
        init_state_c.h(nqpm+ancillary_bits-1-i)

    backend = AerSimulator()

    matAdag = numpy.zeros((d0, d0), dtype=complex)
    for i in range(d0-1):
        matAdag[i+1, i] = numpy.sqrt(i+1)
    matA = matAdag.T

    electrons = generate_electrons(matA, matAdag, 4, nqpm)

    gate_electrons = UnitaryGate(electrons, label="electrons")
    init_state_c.append(gate_electrons, qubit_list)

    tomo_state = StateTomography(init_state_c, measurement_indices=qubit_list)

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    print("Finding husimi...")
    fids = husimi_with_duplicates(50,50, dens_mat_state, nqpm, max_value = max_value)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    return fids


def qifs_gaussian(num_shots = 1024, nqpm = 4, max_value = 2*numpy.pi):
    init_state_c = init_gaussian(nqpm)
    backend = AerSimulator()

    tomo_state = StateTomography(init_state_c, measurement_indices=[i for i in range(0, nqpm)])

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    print("Finding husimi...")
    fids = simple_husimi(50,50, dens_mat_state, nqpm, max_value = max_value)
    
    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    return fids

def qifs_ideal(num_shots = 1024, nqpm = 4):
    init_state_c = init_circle_aer(nqpm, 2)
    backend = AerSimulator()

    tomo_state = StateTomography(init_state_c, measurement_indices=[i for i in range(0, nqpm)])

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    print("Finding husimi...")
    fids = simple_husimi(50,50, dens_mat_state, nqpm, max_value = 6*numpy.pi)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    return fids

def qifs(num_shots=128, nqpm = 4, occupied_states = 2, num_states = 10):
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
    #qifs_gaussian()
    #qifs_ideal()
    qifs_check(num_shots = 1024)
    #qifs_animation(nqpm = 4)
    #qifs(num_states = 20)
