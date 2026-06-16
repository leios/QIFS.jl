'''
    husimi.py
    modification of wigner.py for husimi

    Notes: I decided to go with two different qumoderegisters instead of one
           with num_qumodes = 2 because it seemed like it would be faster. I can
           independently manipulate one state and use the other for measurement.

           if num_qumodes = 2, then plot w/
               bosonic_qiskit.wigner.plot_wigner(circuit0, state0, qmr0[1])
               bosonic_qiskit.wigner.plot_wigner(circuit0, state0, qmr0[0])

'''

import bosonic_qiskit
import qiskit
import numpy
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from math import pi

'''
# TODO
def append_qifs_square(circuit, qmr):
'''

def find_fidelity(state0, circuit0, state1, circuit1):
    # This errors out for some reason...
    dens_matrix0 = bosonic_qiskit.util.trace_out_qubits(circuit0, state0)
    dens_matrix1 = bosonic_qiskit.util.trace_out_qubits(circuit1, state1)
    return qiskit.quantum_info.state_fidelity(dens_matrix0, dens_matrix1)
    # so... full matrix for now.
    #full_dens_mat0 = qiskit.quantum_info.DensityMatrix(state0)
    #full_dens_mat1 = qiskit.quantum_info.DensityMatrix(state1)
    #dens_mat0 = qiskit.quantum_info.partial_trace(full_dens_mat0, [1])
    #dens_mat1 = qiskit.quantum_info.partial_trace(full_dens_mat1, [1])
    #fidelity = qiskit.quantum_info.state_fidelity(dens_mat0, dens_mat1)
    #print(fidelity)

def make_displacement_circuit(x, y, xmax, ymax, qmr, qr):
    x_ratio = 2*numpy.pi*x/xmax
    y_ratio = 2*numpy.pi*y/xmax
    move_ref_circuit = bosonic_qiskit.CVCircuit(qmr, qr)
    move_ref_circuit.cv_d(-numpy.pi+x_ratio+(-numpy.pi+y_ratio)*1j,qmr[0])
    return move_ref_circuit

def simple_husimi(xmax, ymax, qmr0, qr0, qmr1, qr1, state0, circuit0):
    fids = numpy.zeros((xmax, ymax))
    for i in range(0,xmax):
        for j in range(0,ymax):
            move_ref_circuit = make_displacement_circuit(i, j, xmax, ymax, qmr1, qr1)
            state1, _, _ = bosonic_qiskit.util.simulate(move_ref_circuit)
            fids[i][j] = find_fidelity(state0, circuit0, state1, move_ref_circuit)
            #bosonic_qiskit.wigner.plot_wigner(circuit0, state0, qmr0[0])
            #bosonic_qiskit.wigner.plot_wigner(move_ref_circuit, state1, qmr1[0])
    return fids

def qifs():
    qmr0 = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=6)
    qr0 = qiskit.QuantumRegister(size=1)
    
    qmr1 = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=6)
    qr1 = qiskit.QuantumRegister(size=1)
    
    circuit0 = bosonic_qiskit.CVCircuit(qmr0, qr0)
    
    init_ref_circuit = bosonic_qiskit.CVCircuit(qmr1, qr1)
    
    # Initialize your qubit (should have no effect on Fock state Wigner function)
    circuit0.initialize([1,0], qr0[0])
    
    init_ref_circuit.initialize([1,0], qr1[0])
    
    # Initialize the qumode to a zero Fock sate
    circuit0.cv_initialize(0, qmr0[0])
    init_ref_circuit.cv_initialize(0, qmr1[0])
    
    # append_qifs_square(circuit0, qmr0)
    # append_affine_rotate(circuit0, qmr0)
    
    state0, _, _ = bosonic_qiskit.util.simulate(circuit0)
    state1, _, _ = bosonic_qiskit.util.simulate(init_ref_circuit)
    #print(state0)
    #print(state1)

    fids = simple_husimi(20, 20, qmr0, qr0, qmr1, qr1, state0, circuit0)
    plt.imshow(fids, cmap='hot', interpolation='nearest')
    plt.show()

qifs()
