import numpy as np
from qiskit import QuantumCircuit
from scipy.linalg import expm, sqrtm
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Kraus, DensityMatrix, SuperOp

import matplotlib.pyplot as plt

def generate_mat_displacement(matA,matAdag,alpha):
    matD = expm(alpha/np.sqrt(2) * matAdag - np.conj(alpha)/np.sqrt(2) * matA)
    return matD

def log_binomial(a,b):

    result_denominator = 0.0
    result_numerator = 0.0
    b1 = min(b,a-b)

    for n in range(b1):
        result_denominator += np.log(n+1)
        result_numerator += np.log(a-n)

    return result_numerator - result_denominator

def get_superoperator_expansion(d,scaling):
    
    if scaling == 1:
        ValueError("scaling parameter is 1, and the state will not be expanded. Consider not using this or 0.9999...")
    elif scaling > 1:
        ValueError("this expansion channel works for scaling parameter < 1.")

    mat_superop = np.zeros((d**2, d**2), dtype=float)

    for s in range(d):
        for k in range(d):
            # ind1 = k*d + s
            ind1 = k + s*d # index = row + (col - 1) * d
            for j in range(min(d-k,d-s)):
                # ind2 = (k+j)*d + s+j
                ind2 = k+j + (s+j)*d
                log_coeff_ket = log_binomial(k+j,k)/2 + (k+1)*np.log(scaling) + (j/2)*np.log(1-scaling**2)
                log_coeff_bra = log_binomial(s+j,s)/2 + (s+1)*np.log(scaling) + (j/2)*np.log(1-scaling**2)
                mat_superop[ind2,ind1] += np.exp(log_coeff_ket)*np.exp(log_coeff_bra)

    return mat_superop

N = 5 # number of qubits
d = 2**N
num_shots = 5000
M = 0 #3 # size of circle, less than N
size_q = 20
scaling = 0.5 # (x, p) -> (x/scaling, p/scaling)

# for displacement operator
matAdag = np.zeros((d, d), dtype=complex)
for i in range(d-1):
    matAdag[i+1, i] = np.sqrt(i+1)
matA = matAdag.T

# for dephasing
dephasing_ops = [
    np.array([[1, 0], [0, 0]]),
    np.array([[0, 0], [0, 1]])
]
dephasing_channel = Kraus(dephasing_ops)

# for expansion
superop_matrix_scaling = get_superoperator_expansion(d, scaling)
channel_scaling = SuperOp(superop_matrix_scaling)
kraus_matrix_scaling = Kraus(channel_scaling).data

# recover TP
identity = np.eye(d)
sum_K_dagger_K = sum(np.conjugate(K.T) @ K for K in kraus_matrix_scaling)
leakage_matrix = identity - sum_K_dagger_K
K_dump = sqrtm(leakage_matrix)
kraus_list_scaling_cptp = list(kraus_matrix_scaling) + [K_dump]
channel_scaling_cptp = Kraus(kraus_list_scaling_cptp)

print("leaking:",np.trace(leakage_matrix)/d)

# print("Is CPTP?:", channel_scaling_cptp.is_cptp())  # CPTP
# print("Is CP?:",   channel_scaling_cptp.is_cp())    # Complete Positivity
# print("Is TP?:",   channel_scaling_cptp.is_tp())    # Trace Preserving

# create a circuit with N qubits and N bits
simulator = AerSimulator()

q_range = np.linspace(-7, 7, size_q) # equal or less than 2^N
p_range = np.linspace(-7, 7, size_q)
list_all_qubits = list(range(N))
zero_key = '0' * N
data_husimi = np.zeros((size_q,size_q), dtype=float)

# qc = QuantumCircuit(N, N) # classical bits for registering measurement results
# qc.x(N-1)
# for n in range(M):
#     qc.h(n)
#     qc.append(dephasing_channel.to_instruction(), [n])

# rho = DensityMatrix(qc)
# matrix_data = rho.data
# print(matrix_data.real)

for i, q in enumerate(q_range):

    print(i)

    for j, p in enumerate(p_range):
        
        qc = QuantumCircuit(N, N) # classical bits for registering measurement results
        # qc.x(N-1)
        for n in range(M):
            qc.h(n)
            qc.append(dephasing_channel.to_instruction(), [n])
        qc.append(channel_scaling_cptp.to_instruction(), list_all_qubits)
        
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
