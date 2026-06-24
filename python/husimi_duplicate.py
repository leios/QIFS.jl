import numpy as np
from qiskit import QuantumCircuit
from scipy.linalg import expm, block_diag
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Kraus, DensityMatrix

import matplotlib.pyplot as plt

def generate_mat_displacement(matA,matAdag,alpha):
    matD = expm(alpha/np.sqrt(2) * matAdag - np.conj(alpha)/np.sqrt(2) * matA)
    return matD

def generate_mat_diplication(matA,matAdag,alpha1,alpha2):
    matD1 = generate_mat_displacement(matA,matAdag,alpha1)
    matD2 = generate_mat_displacement(matA,matAdag,alpha2)
    return block_diag(matD1,matD2)

dephasing_ops = [
    np.array([[1, 0], [0, 0]]),
    np.array([[0, 0], [0, 1]])
]
dephasing_channel = Kraus(dephasing_ops)

N0 = 5
N1 = 1
N = N0+N1 # number of qubits

# d = 2**N
d0 = 2**N0

num_shots = 5000
M = 3 # size of circle, less than N
size_q = 20

# for displacement operator
matAdag = np.zeros((d0, d0), dtype=complex)
for i in range(d0-1):
    matAdag[i+1, i] = np.sqrt(i+1)
matA = matAdag.T

simulator = AerSimulator()

q_range = np.linspace(-7, 7, size_q) # equal or less than 2^N
p_range = np.linspace(-7, 7, size_q)

list_all_qubits = list(range(N))
# list_all_qubits_of_interest = list(range(N0))
zero_key = '0'*N
zero_except_last_key = '1'*N1 + '0'*N0 # qubit are arranged backward in qiskit
data_husimi = np.zeros((size_q,size_q), dtype=float)
# data_husimi_0 = np.zeros((size_q,size_q), dtype=float)
# data_husimi_1 = np.zeros((size_q,size_q), dtype=float)

for i, q in enumerate(q_range):

    print(i)

    for j, p in enumerate(p_range):
        
        qc = QuantumCircuit(N, N) # classical bits for registering measurement results
        for n in range(M):
            qc.h(n)
            qc.append(dephasing_channel.to_instruction(), [n])
        qc.h(N-1)
        qc.append(dephasing_channel.to_instruction(), [N-1])
        
        mat_diplication = generate_mat_diplication(matA,matAdag,-4.0,4.0+1j)
        gate_diplication = UnitaryGate(mat_diplication, label="diplication")
        qc.append(gate_diplication, list_all_qubits)

        # mat_displacement = generate_mat_displacement(matA,matAdag,-q-1j*p) # D(-alpha)
        mat_displacement = generate_mat_diplication(matA,matAdag,-q-1j*p,-q-1j*p)
        gate_displacement = UnitaryGate(mat_displacement, label="displacement_operator")
        qc.append(gate_displacement, list_all_qubits)
        
        # measure all qubits
        for n in range(N):
            qc.measure(n, n)
        # qc.measure_all()

        counts = simulator.run(qc, shots=num_shots).result().get_counts()
        data_husimi[i,j] = (counts.get(zero_key, 0)+counts.get(zero_except_last_key, 0))/num_shots
        # data_husimi_0[i,j] = counts.get(zero_key, 0)/num_shots
        # data_husimi_1[i,j] = counts.get(zero_except_last_key, 0)/num_shots

# plt.plot(data_husimi[:,int(d/2)])
fig, ax = plt.subplots(figsize=(6, 5))
contour = ax.contourf(q_range, p_range, data_husimi.T, levels=100)
fig.colorbar(contour)
plt.show()

