import matplotlib.pyplot as plt
import bosonic_qiskit
import qiskit
import numpy
import random
from scipy.linalg import expm, block_diag
from scipy.linalg import sqrtm
from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import DensityMatrix, state_fidelity, Kraus, Stinespring, Operator
from qiskit_experiments.library import StateTomography

def make_displacement_circuit(x, y, xmax, ymax, qmr, max_value = 2*numpy.pi):
    x_ratio = max_value*x/xmax
    y_ratio = max_value*y/xmax
    move_ref_circuit = bosonic_qiskit.CVCircuit(qmr)
    move_ref_circuit.cv_d(-0.5*max_value+x_ratio+(-0.5*max_value+y_ratio)*1j,qmr[0])
    return move_ref_circuit

def append_displacement_circuit(x, y, xmax, ymax, qmr, circuit):
    x_ratio = 2*numpy.pi*x/xmax
    y_ratio = 2*numpy.pi*y/xmax
    circuit.cv_d(-numpy.pi+x_ratio+(-numpy.pi+y_ratio)*1j,qmr[0])

# maybe partial trace?
def simple_husimi(xmax, ymax, backend, dens_mat, nqpm, max_value = 2*numpy.pi):
    fids = numpy.zeros((xmax, ymax))

    (qmr_ref, c_ref) = init_bosonic_gaussian(nqpm)

    state_ref, _, _ = bosonic_qiskit.util.simulate(c_ref)

    for i in range(0,xmax):
        for j in range(0,ymax):
            c_move_ref = make_displacement_circuit(i, j, xmax, ymax, qmr_ref, max_value = max_value)
            state_ref, _, _ = bosonic_qiskit.util.simulate(c_move_ref)

            dens_mat_ref = qiskit.quantum_info.DensityMatrix(state_ref)

            # trace out ancillary bit
            #dens_mat_ref_partial = qiskit.quantum_info.partial_trace(dens_mat_ref, [nqpm])
            #dens_mat1 = qiskit.quantum_info.partial_trace(full_dens_mat1, [1])

            fids[i][j] = state_fidelity(dens_mat, dens_mat_ref)
            #print(fids[i][j])
    return fids

def find_fidelity(mat1, mat2):
    sqrt_mat1 = sqrtm(mat1)
    return numpy.real(numpy.trace(sqrtm(sqrt_mat1 @ mat2 @ sqrt_mat1))**2)

def husimi_with_duplicates(xmax, ymax, backend, dens_mat, nqpm, max_value = 2*numpy.pi, num_ancillary = 3):
    fids = numpy.zeros((xmax, ymax))

    (qmr_ref, c_ref) = init_bosonic_gaussian(nqpm)

    state_ref, _, _ = bosonic_qiskit.util.simulate(c_ref)

    total_objects = num_ancillary**2

    for i in range(0,xmax):
        for j in range(0,ymax):
            c_move_ref = make_displacement_circuit(i, j, xmax, ymax, qmr_ref, max_value = max_value) 
            state_ref, _, _ = bosonic_qiskit.util.simulate(c_move_ref)

            dens_mat_ref = qiskit.quantum_info.DensityMatrix(state_ref)

            for k in range(total_objects-1):
                stride = nqpm**2
                dens_mat_mini = dens_mat.data[k*stride:(k+1)*stride, k*stride:(k+1)*stride]
                new_fid = find_fidelity(dens_mat_mini, dens_mat_ref.data) / total_objects

                fids[i][j] = fids[i][j] + new_fid
    return fids


def find_backend():
    # Noisy simulator
    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")
    return AerSimulator.from_backend(backend)

def init_gaussian(nqpm, ancillary_bits = 0):
    total_bits = nqpm + ancillary_bits
    init_c = QuantumCircuit(total_bits, total_bits)
    return init_c

def init_bosonic_gaussian(nqpm, ancillary_bits = 0):
    qmr = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=nqpm)
    if ancillary_bits > 0:
        qr = qiskit.QuantumRegister(size=ancillary_bits)
        init_c = bosonic_qiskit.CVCircuit(qmr, qr)

        # init fock state at |0>
        init_c.cv_initialize(0, qmr[0])

        for i in range(ancillary_bits):
            init_c.initialize([1,0], qr[i])

        return (qmr, qr, init_c)
    else:
        init_c = bosonic_qiskit.CVCircuit(qmr)


        # init fock state at |0>
        init_c.cv_initialize(0, qmr[0])

        return (qmr, init_c)


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

# removing unnecessary resets from circuit after decomposition
def decompose_init_circuit(init_c):
    # ch = "circuit hardware"
    init_ch = init_c.decompose().decompose()
    init_ch.data = [
        inst for inst in init_ch.data
        if inst.operation.name != 'reset'
    ]
    return init_ch

def generate_mat_displacement(matA,matAdag,alpha):
    matD = expm(alpha/numpy.sqrt(2) * matAdag - numpy.conj(alpha)/numpy.sqrt(2) * matA)
    return matD

def rotate_in_phase_space(val, theta):
    re = val.real*numpy.cos(theta) - val.imag*numpy.sin(theta)
    im = val.real*numpy.sin(theta) + val.imag*numpy.cos(theta)
    return re + im*1j

def generate_electrons(matA, matAdag, radius, nqpm):

    # top and bottom
    matD1 = numpy.eye(2**nqpm)
    matD2 = numpy.eye(2**nqpm)
    #matD1 = generate_mat_displacement(matA,matAdag, radius*1j)
    #matD2 = generate_mat_displacement(matA,matAdag,-radius*1j)

    # one arc
    matD3 = numpy.eye(2**nqpm)
    matD4 = numpy.eye(2**nqpm)
    #matD3 = generate_mat_displacement(matA, matAdag, rotate_in_phase_space(radius*1j, numpy.pi/3))
    #matD4 = generate_mat_displacement(matA,matAdag, rotate_in_phase_space(radius*1j, -2*numpy.pi/3))

    # the other arc
    matD5 = numpy.eye(2**nqpm)
    matD6 = numpy.eye(2**nqpm)
    #matD5 = generate_mat_displacement(matA,matAdag, rotate_in_phase_space(radius*1j, 2*numpy.pi/3))
    #matD6 = generate_mat_displacement(matA,matAdag, rotate_in_phase_space(radius*1j, -numpy.pi/3))

    # both nucleii
    matD7 = numpy.eye(2**nqpm)
    matD8 = numpy.eye(2**nqpm)
    return block_diag(matD1,matD2, matD3, matD4, matD5, matD6, matD7, matD8)

def qifs_animation(num_shots = 1024, nqpm = 4, max_value = 2*numpy.pi):

    d0 = 2**nqpm

    ancillary_bits = 3

    qubit_list = list(range(nqpm + ancillary_bits))

    #state_c = init_gaussian(nqpm, ancillary_bits = ancillary_bits)
    state_c = init_circle_aer(nqpm+ancillary_bits, 2)
    backend = AerSimulator()

    matAdag = numpy.zeros((d0, d0), dtype=complex)
    for i in range(d0-1):
        matAdag[i+1, i] = numpy.sqrt(i+1)
    matA = matAdag.T

    electrons = generate_electrons(matA, matAdag, 2, nqpm)

    gate_electrons = UnitaryGate(electrons, label="electrons")
    state_c.append(gate_electrons, qubit_list)

    #for n in range(nqpm + ancillary_bits):
    #    state_c.measure(n, n)

    tomo_state = StateTomography(state_c, measurement_indices=qubit_list)

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    print("Finding husimi...")
    fids = husimi_with_duplicates(50,50, backend, dens_mat_state, nqpm, max_value = max_value)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    return fids


def qifs_gaussian(num_shots = 1024, nqpm = 4, max_value = 2*numpy.pi):
    qmr, init_state_c = init_gaussian(nqpm)
    init_state_c = decompose_init_circuit(init_state_c)
    backend = AerSimulator()

    tomo_state = StateTomography(init_state_c, measurement_indices=[i for i in range(0, nqpm)])

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    print("Finding husimi...")
    fids = simple_husimi(50,50, backend, dens_mat_state, nqpm, max_value = max_value)
    
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
    fids = simple_husimi(50,50, backend, dens_mat_state, nqpm, max_value = 6*numpy.pi)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    return fids

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
    fids = simple_husimi(50,50, backend, dens_mat_state, nqpm)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    #ideal_backend = AerSimulator()
    #tomo_state_ideal = StateTomography(init_state_c, measurement_indices=[0])

    #dens_mat_state_ideal = tomo_state_ideal.run(ideal_backend, shots = num_shots).block_for_results().analysis_results("state", dataframe = True).iloc[0].value
    #fids = simple_husimi(5, 5, backend, dens_mat_state)

    return fids

if __name__ == "__main__":
    #qifs(num_states = 20)
    #qifs_gaussian(nqpm = 4)
    qifs_animation(nqpm = 4)
    #qifs_ideal(nqpm = 4)
