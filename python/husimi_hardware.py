import matplotlib.pyplot as plt
import bosonic_qiskit
import qiskit
import numpy
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit_experiments.library import StateTomography

def make_displacement_circuit(x, y, xmax, ymax, qmr, qr):
    x_ratio = 2*numpy.pi*x/xmax
    y_ratio = 2*numpy.pi*y/xmax
    move_ref_circuit = bosonic_qiskit.CVCircuit(qmr, qr)
    move_ref_circuit.cv_d(-numpy.pi+x_ratio+(-numpy.pi+y_ratio)*1j,qmr[0])
    return move_ref_circuit

def append_displacement_circuit(x, y, xmax, ymax, qmr, circuit):
    x_ratio = 2*numpy.pi*x/xmax
    y_ratio = 2*numpy.pi*y/xmax
    circuit.cv_d(-numpy.pi+x_ratio+(-numpy.pi+y_ratio)*1j,qmr[0])

# maybe partial trace?
def simple_husimi(xmax, ymax, backend, dens_mat, nqpm):
    fids = numpy.zeros((xmax, ymax))

    (qmr_ref, qr_ref, c_ref) = init_gaussian(nqpm)

    state_ref, _, _ = bosonic_qiskit.util.simulate(c_ref)

    for i in range(0,xmax):
        for j in range(0,ymax):
            c_move_ref = make_displacement_circuit(i, j, xmax, ymax, qmr_ref, qr_ref)
            state_ref, _, _ = bosonic_qiskit.util.simulate(c_move_ref)

            dens_mat_ref = qiskit.quantum_info.DensityMatrix(state_ref)

            # trace out ancillary bit
            dens_mat_ref_partial = qiskit.quantum_info.partial_trace(dens_mat_ref, [nqpm])
            #dens_mat1 = qiskit.quantum_info.partial_trace(full_dens_mat1, [1])

            fids[i][j] = state_fidelity(dens_mat, dens_mat_ref_partial)
            #print(fids[i][j])
    return fids

def find_backend():
    # Noisy simulator
    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")
    return AerSimulator.from_backend(backend)

def init_gaussian(nqpm):
    qmr = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=nqpm)
    qr = qiskit.QuantumRegister(size=1)

    init_c = bosonic_qiskit.CVCircuit(qmr, qr)

    # init register at 1 + 0im
    init_c.initialize([1,0], qr[0])

    # init fock state at |0>
    init_c.cv_initialize(0, qmr[0])

    return (qmr, qr, init_c)

# N Number of qubits
# M size of circle M < N
def init_circle(N, M):

    dephasing_ops = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]])
    ]
    dephasing_channel = Kraus(dephasing_ops)

    

# removing unnecessary resets from circuit after decomposition
def decompose_init_circuit(init_c):
    # ch = "circuit hardware"
    init_ch = init_c.decompose().decompose()
    init_ch.data = [
        inst for inst in init_ch.data
        if inst.operation.name != 'reset'
    ]
    return init_ch

def qifs(num_shots=1024, nqpm = 4):
    (qmr0, qr0, init_state_c) = init_gaussian(nqpm)

    init_state_ch = decompose_init_circuit(init_state_c)

    backend = find_backend()

    tomo_state = StateTomography(init_state_c, measurement_indices=[i for i in range(0, nqpm)])

    result_state = tomo_state.run(backend, shots = num_shots).block_for_results()

    dens_mat_state = result_state.analysis_results("state", dataframe = True).iloc[0].value

    fids = simple_husimi(50,50, backend, dens_mat_state, nqpm)

    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

    #ideal_backend = AerSimulator()
    #tomo_state_ideal = StateTomography(init_state_c, measurement_indices=[0])

    #dens_mat_state_ideal = tomo_state_ideal.run(ideal_backend, shots = num_shots).block_for_results().analysis_results("state", dataframe = True).iloc[0].value
    #fids = simple_husimi(5, 5, backend, dens_mat_state)

qifs()
