import numpy as np
from qiskit import QuantumCircuit
from scipy.linalg import expm
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Kraus, DensityMatrix

import matplotlib.pyplot as plt

def generate_mat_displacement(matA,matAdag,alpha):
    matD = expm(alpha/np.sqrt(2) * matAdag - np.conj(alpha)/np.sqrt(2) * matA)
    return matD

def generate_mat_diplicate(matA,matAdag,alpha):
    matD = generate_mat_displacement(matA,matAdag,alpha)
    mat0 = np.array([[1, 0], [0, 0]])
    return matD

dephasing_ops = [
    np.array([[1, 0], [0, 0]]),
    np.array([[0, 0], [0, 1]])
]
dephasing_channel = Kraus(dephasing_ops)

N = 4 # number of qubits
d = 2**N
num_shots = 5000
M = 1 # size of circle, less than N
size_q = 20

# for displacement operator
matAdag = np.zeros((d, d), dtype=complex)
for i in range(d-1):
    matAdag[i+1, i] = np.sqrt(i+1)
matA = matAdag.T

# create a circuit with N qubits and N bits
simulator = AerSimulator()

q_range = np.linspace(-7, 7, size_q) # equal or less than 2^N
p_range = np.linspace(-7, 7, size_q)
list_all_qubits = list(range(N))
zero_key = '0' * N
data_husimi = np.zeros((size_q,size_q), dtype=float)

qc = QuantumCircuit(N, N) # classical bits for registering measurement results
for n in range(M):
    qc.h(n)
    qc.append(dephasing_channel.to_instruction(), [n])
qc.h(N-1)
qc.append(dephasing_channel.to_instruction(), [N-1])
rho = DensityMatrix(qc)
matrix_data = rho.data
print(matrix_data)

for i, q in enumerate(q_range):

    print(i)

    for j, p in enumerate(p_range):
        
        qc = QuantumCircuit(N, N) # classical bits for registering measurement results
        for n in range(M):
            qc.h(n)
            qc.append(dephasing_channel.to_instruction(), [n])
        
        mat_displacement = generate_mat_displacement(matA,matAdag,-q-1j*p) # D(-alpha)
        gate_displacement = UnitaryGate(mat_displacement, label="displacement_operator")
        qc.append(gate_displacement, list_all_qubits)
        
        # measure all qubits
        for n in range(N):
            qc.measure(n, n)
        # qc.measure_all()

        counts = simulator.run(qc, shots=num_shots).result().get_counts()
        data_husimi[i,j] = counts.get(zero_key, 0)/num_shots

# plt.plot(data_husimi[:,int(d/2)])
fig, ax = plt.subplots(figsize=(6, 5))
contour = ax.contourf(q_range, p_range, data_husimi.T, levels=100)
fig.colorbar(contour)
plt.show()
