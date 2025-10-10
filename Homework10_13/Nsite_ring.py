import numpy as np
from numpy import kron
from scipy.linalg import eigh
import cmath

# ----- Preparation -----
N = 6
g = 1.0
J = 1.0
h = 1.2

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
id2 = np.eye(2, dtype=complex)

def kron_n(op_list):
    res = np.array([[1]], dtype=complex)
    for op in op_list:
        res = kron(res, op)
    return res

# ------ Hamiltonian Construction -----
dim = 2**N
H = np.zeros((dim, dim), dtype=complex)

for j in range(N):
    jm1 = (j - 1) % N
    jp1 = (j + 1) % N

    op_zyz = []
    for site in range(N):
        if site == jm1:
            op_zyz.append(sz)
        elif site == j:
            op_zyz.append(sy)
        elif site == jp1:
            op_zyz.append(sz)
        else:
            op_zyz.append(id2)
    H -= g * kron_n(op_zyz)

    op_zz = []
    for site in range(N):
        if site == j:
            op_zz.append(sz)
        elif site == jp1:
            op_zz.append(sz)
        else:
            op_zz.append(id2)
    H -= J * kron_n(op_zz)

    op_x = []
    for site in range(N):
        if site == j:
            op_x.append(sx)
        else:
            op_x.append(id2)
    H -= h * kron_n(op_x)

# ----- Translation Operator Construction -----
T = np.zeros((dim, dim), dtype=complex)

for i in range(dim):
    config = [(i >> bit) & 1 for bit in range(N)]
    translated_config = config[-1:] + config[:-1]
    
    j = sum(bit << pos for pos, bit in enumerate(translated_config))
    T[j, i] = 1.0

# ----- Decomposition -----
eigvals_T, eigvecs_T = np.linalg.eig(T)

momentum_sectors = {}
for ki in range(N):
    momentum_sectors[ki] = []

for i, eigenvalue in enumerate(eigvals_T):
    phase = cmath.phase(eigenvalue)
    if phase < 0:
        phase += 2 * np.pi
    k_index = round(phase * N / (2 * np.pi)) % N
    momentum_sectors[k_index].append(i)

print(f"Case: g={g} J={J} h={h} Ns={N}\n")
print("(1) The eigenvalues in each Momentum_Sector_ki:\n")

ground_energy = float('inf')
ground_state = None

for k_index in range(N):
    indices = momentum_sectors[k_index]
    if not indices:
        continue
        
    Vk = eigvecs_T[:, indices]
    Hk = Vk.conj().T @ H @ Vk
    Ek, psi_k = eigh(Hk)
    
    formatted_eigenvalues = "[" + " ".join([f"{val:.8f}" for val in Ek.real]) + "]"
    print(f"ki= {k_index} : {formatted_eigenvalues}")
    
    # Find the global ground state
    if Ek[0] < ground_energy:
        ground_energy = Ek[0]
        ground_state = Vk @ psi_k[:, 0]

# ----- Ground State Properties -----
E_per_site = ground_energy / N

sigmaz_exp = 0+0j
sigmax_exp = 0+0j

for j in range(N):
    
    ops_x = [id2] * N
    ops_z = [id2] * N
    ops_x[j] = sx
    ops_z[j] = sz
    
    op_x_j = kron_n(ops_x)
    op_z_j = kron_n(ops_z)

    sigmax_exp += np.vdot(ground_state, op_x_j @ ground_state)
    sigmaz_exp += np.vdot(ground_state, op_z_j @ ground_state)

sigmax_exp /= N
sigmaz_exp /= N

threshold = 1e-10

# ----- Output Formatting -----
def Format_complex(z, threshold=1e-10):
    real_part = z.real
    imag_part = z.imag
    
    if abs(real_part) < threshold:
        real_part = 0.0
    if abs(imag_part) < threshold:
        imag_part = 0.0

    return f"({real_part} + {imag_part}j)"

print(f"\n(2) Ground_State_energy_per_site= {E_per_site}")
print(f"(2) Ground_State_sigmax_per_site= {Format_complex(sigmax_exp, threshold)}")
print(f"(2) Ground_State_sigmaz_per_site= {Format_complex(sigmaz_exp, threshold)}")