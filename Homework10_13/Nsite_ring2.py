import numpy as np
from numpy import kron
from scipy.linalg import eigh
import cmath

# Parameters 
N = 8
g = 1.0
J = 1.0
h = 1.0

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
id2 = np.eye(2, dtype=complex)


def kron_n(op_list):
    """
    Compute Kronecker product of a list of operators.
    """
    res = np.array([[1]], dtype=complex)
    for op in op_list:
        res = kron(res, op)
    return res


def get_representative_info(config_int, N):
    """
    Find representative configuration and translation steps.
    """
    current = config_int
    min_config = config_int
    min_steps = 0

    for steps in range(1, N):
        current = ((current >> 1) | ((current & 1) << (N-1))) & ((1 << N) - 1)
        if current < min_config:
            min_config = current
            min_steps = steps
    
    is_rep = 1 if (config_int == min_config) else 0
    return is_rep, min_config, min_steps


def build_check_table(N):
    """
    Build the Check table as shown in slides.
    """
    dim = 1 << N  
    check_table = np.zeros((dim, 3), dtype=int)
    
    for i in range(dim):
        is_rep, rep, steps = get_representative_info(i, N)
        check_table[i] = [is_rep, rep, steps]
    
    return check_table


def projection_operator(k, rep_config, N, check_table):
    """
    Apply projection operator P_k to representative configuration.
    """
    dim = 1 << N
    result_state = np.zeros(dim, dtype=complex)
    
    # Get all configurations in the translation orbit
    for config_int in range(dim):
        if check_table[config_int, 1] == rep_config:
            steps = check_table[config_int, 2]
            phase = cmath.exp(1j * 2 * np.pi * k * steps / N)
            result_state[config_int] = phase
    
    result_state /= N
    norm_sq = np.vdot(result_state, result_state).real
    
    return result_state, norm_sq


def build_hamiltonian_matrix(N, g, J, h):
    """
    Build the full Hamiltonian.
    """
    dim = 1 << N
    H = np.zeros((dim, dim), dtype=complex)
    
    def create_local_operator(site_index, op_matrix):
        op_list = [id2] * N
        op_list[site_index] = op_matrix
        return kron_n(op_list)
    
    for site in range(N):

        site_m1 = (site - 1) % N
        site_p1 = (site + 1) % N
        
        op_z1 = create_local_operator(site_m1, sz)
        op_y = create_local_operator(site, sy)
        op_z2 = create_local_operator(site_p1, sz)
        H -= g * (op_z1 @ op_y @ op_z2)
        
        op_z_j = create_local_operator(site, sz)
        op_z_jp1 = create_local_operator(site_p1, sz)
        H -= J * (op_z_j @ op_z_jp1)
        
        op_x = create_local_operator(site, sx)
        H -= h * op_x
    
    return H


def main():
    
    check_table = build_check_table(N)
    H_full = build_hamiltonian_matrix(N, g, J, h)
    dim = 1 << N
    
    print(f"Case: g={g} J={J} h={h} Ns={N}\n")
    print("(1) The eigenvalues in each Momentum_Sector_ki:\n")
    
    # Find all representative configurations
    representatives = []
    for i in range(dim):
        if check_table[i, 0] == 1:  
            representatives.append(i)
    
    # For each momentum sector, build the basis and diagonalize Hamiltonian
    ground_energy = float('inf')
    ground_state_full = None
    
    for k in range(N):
        basis_states = []
        norms = []
        
        # Build momentum basis for this k sector
        for rep in representatives:
            state, norm = projection_operator(k, rep, N, check_table)
            if state is not None:  
                basis_states.append(state)
                norms.append(norm)
        
        if not basis_states:
            print(f"ki= {k} : []")
            continue
            
        # Normalize basis states
        normalized_basis = []
        for state, norm in zip(basis_states, norms):
            normalized_state = state / np.sqrt(norm)
            normalized_basis.append(normalized_state)
        
        # Build Hamiltonian in this momentum sector
        size = len(normalized_basis)
        H_k = np.zeros((size, size), dtype=complex)
        
        for i in range(size):
            for j in range(size):
                H_k[i, j] = np.vdot(normalized_basis[i], H_full @ normalized_basis[j])
        
        # Diagonalize
        eigenvalues, eigenvectors = eigh(H_k)
        
        # Format output as in sample
        formatted_eigenvalues = "[" + " ".join([f"{val:.8f}" for val in eigenvalues.real]) + "]"
        print(f"ki= {k} : {formatted_eigenvalues}")
        
        # Find ground state
        if eigenvalues[0] < ground_energy:
            ground_energy = eigenvalues[0]
            
            # Reconstruct ground state in full Hilbert space
            gs_in_sector = eigenvectors[:, 0]
            ground_state_full = np.zeros(dim, dtype=complex)
            for idx, coeff in enumerate(gs_in_sector):
                ground_state_full += coeff * normalized_basis[idx]
    

    # Calculate ground state properties

    E_per_site = ground_energy / N
    
    sigmaz_exp = 0 + 0j
    sigmax_exp = 0 + 0j
    
    for site in range(N):
        op_list_x = [id2] * N
        op_list_x[site] = sx
        op_x = kron_n(op_list_x)
         
        op_list_z = [id2] * N
        op_list_z[site] = sz
        op_z = kron_n(op_list_z)
        
        sigmax_exp += np.vdot(ground_state_full, op_x @ ground_state_full)
        sigmaz_exp += np.vdot(ground_state_full, op_z @ ground_state_full)
    
    sigmax_exp /= N
    sigmaz_exp /= N
    

    # Output results 

    threshold = 1e-10
    
    def format_complex(z, threshold=1e-10):
        real_part = z.real
        imag_part = z.imag
        
        if abs(real_part) < threshold:
            real_part = 0.0
        if abs(imag_part) < threshold:
            imag_part = 0.0
        
        return real_part, imag_part
    
    sigmax_real, sigmax_imag = format_complex(sigmax_exp, threshold)
    sigmaz_real, sigmaz_imag = format_complex(sigmaz_exp, threshold)
    
    print(f"\n(2) Ground_State_energy_per_site= {E_per_site}")
    print(f"(2) Ground_State_sigmax_per_site= ({sigmax_real}+{sigmax_imag}j)")
    print(f"(2) Ground_State_sigmaz_per_site= ({sigmaz_real}+{sigmaz_imag}j)")

if __name__ == "__main__":
    main()