import matplotlib.pyplot as plt
import bosonic_qiskit
import qiskit
import numpy
import random
from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import DensityMatrix, state_fidelity, Kraus, Stinespring, Operator
from qiskit_experiments.library import StateTomography

def make_displacement_circuit(x, y, xmax, ymax, qmr):
    x_ratio = 2*numpy.pi*x/xmax
    y_ratio = 2*numpy.pi*y/xmax
    move_ref_circuit = bosonic_qiskit.CVCircuit(qmr)
    move_ref_circuit.cv_d(-numpy.pi+x_ratio+(-numpy.pi+y_ratio)*1j,qmr[0])
    return move_ref_circuit

def append_displacement_circuit(x, y, xmax, ymax, qmr, circuit):
    x_ratio = 2*numpy.pi*x/xmax
    y_ratio = 2*numpy.pi*y/xmax
    circuit.cv_d(-numpy.pi+x_ratio+(-numpy.pi+y_ratio)*1j,qmr[0])

# maybe partial trace?
def simple_husimi(xmax, ymax, backend, dens_mat, nqpm):
    fids = numpy.zeros((xmax, ymax))

    (qmr_ref, c_ref) = init_gaussian(nqpm)

    state_ref, _, _ = bosonic_qiskit.util.simulate(c_ref)

    for i in range(0,xmax):
        for j in range(0,ymax):
            c_move_ref = make_displacement_circuit(i, j, xmax, ymax, qmr_ref)
            state_ref, _, _ = bosonic_qiskit.util.simulate(c_move_ref)

            dens_mat_ref = qiskit.quantum_info.DensityMatrix(state_ref)

            # trace out ancillary bit
            #dens_mat_ref_partial = qiskit.quantum_info.partial_trace(dens_mat_ref, [nqpm])
            #dens_mat1 = qiskit.quantum_info.partial_trace(full_dens_mat1, [1])

            fids[i][j] = state_fidelity(dens_mat, dens_mat_ref)
            #print(fids[i][j])
    return fids

def find_backend():
    # Noisy simulator
    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")
    return AerSimulator.from_backend(backend)

def init_gaussian(nqpm):
    qmr = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=nqpm)
    init_c = bosonic_qiskit.CVCircuit(qmr)

    # init fock state at |0>
    init_c.cv_initialize(0, qmr[0])

    return (qmr, init_c)

# swirling
def init_circle_set(N, M, num_circuits):
    # Average over random Z rotations
    thetas = numpy.random.uniform(0, 2*numpy.pi, num_circuits)
    #thetas = numpy.linspace(0, 2*numpy.pi, num_circuits)

    circuits = []
    for theta in thetas:
        qc = QuantumCircuit(N)
        for n in range(M):
            qc.h(n)
            choice = random.choice([0,1,2,3])
            if choice == 1:
                qc.rx(theta, n)
            elif choice == 2:
                qc.ry(theta, n)
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

def qifs_ideal(num_shots = 1024, nqpm = 4):
    init_state_c = init_circle_aer(nqpm, 2)
    backend = AerSimulator()

    tomo_state = StateTomography(init_state_c, measurement_indices=[i for i in range(0, nqpm)])

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    print("Finding husimi...")
    fids = simple_husimi(50,50, backend, dens_mat_state, nqpm)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

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
    fids = simple_husimi(50,50, backend, dens_mat_state, nqpm)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    #ideal_backend = AerSimulator()
    #tomo_state_ideal = StateTomography(init_state_c, measurement_indices=[0])

    #dens_mat_state_ideal = tomo_state_ideal.run(ideal_backend, shots = num_shots).block_for_results().analysis_results("state", dataframe = True).iloc[0].value
    #fids = simple_husimi(5, 5, backend, dens_mat_state)

#qifs()
