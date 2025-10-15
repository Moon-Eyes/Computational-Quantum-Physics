import numpy as np
import itertools

# --- Parameters and Geometry Definition ---
# Parameters from the homework
t = 1.0
U = 8.0
V = 0.4
mu = 4.0

# System geometry
N_SITES = 6
# Hopping pairs for the t term
HOPPING_PAIRS = [(0, 2), (1, 3), (2, 4), (4, 5), (1, 2), (3, 4)]
# Interaction pairs for the V term
INTERACTION_PAIRS = [(0, 1), (2, 3), (1, 4), (3, 5)]

def generate_basis(n_sites, n_particles):
    """Generates basis states for a given number of sites and particles."""
    if n_particles > n_sites or n_particles < 0:
        return []
    basis = []
    for pos in itertools.combinations(range(n_sites), n_particles):
        state = 0
        for i in pos:
            state |= (1 << i)
        basis.append(state)
    return sorted(basis)

def count_set_bits(n):
    """Counts the number of set bits (popcount) in an integer."""
    count = 0
    while n > 0:
        n &= (n - 1)
        count += 1
    return count
    
def get_fermionic_sign(state, i, j):
    """Calculates the fermionic sign for hopping from j to i."""
    if i == j:
        return 1
    # Ensure i < j for consistent mask creation
    if i > j:
        i, j = j, i
    
    # Create a mask for bits between i and j
    mask = (1 << j) - (1 << (i + 1))
    # Count particles between i and j
    p = count_set_bits(state & mask)
    return (-1)**p


def main():
    """Main function to perform Exact Diagonalization."""
    print("=" * 40)
    print("Exact Diagonalization for Extended Hubbard Model")
    print(f"Parameters: t={t}, U={U}, V={V}, mu={mu}")
    print("=" * 40)

    all_eigenvalues = []
    ground_state_info = {
        'energy': float('inf'),
        'vector': None,
        'basis_up': None,
        'basis_down': None,
        'n_up': -1,
        'n_down': -1
    }

    # Iterate over all 49 (7x7) subspaces
    for n_up in range(N_SITES + 1):
        for n_down in range(N_SITES + 1):
            
            basis_up = generate_basis(N_SITES, n_up)
            basis_down = generate_basis(N_SITES, n_down)
            
            dim_up = len(basis_up)
            dim_down = len(basis_down)
            dim = dim_up * dim_down

            if dim == 0:
                continue

            # Create a map from basis state to index for fast lookup
            map_up = {state: i for i, state in enumerate(basis_up)}
            map_down = {state: i for i, state in enumerate(basis_down)}

            H = np.zeros((dim, dim), dtype=np.float64)

            # Construct the Hamiltonian matrix for the (n_up, n_down) subspace
            for i in range(dim_up):
                for j in range(dim_down):
                    k = i * dim_down + j  # Combined index
                    state_up = basis_up[i]
                    state_down = basis_down[j]

                    # --- Diagonal terms ---
                    # On-site interaction (U term)
                    doubly_occupied_sites = state_up & state_down
                    H[k, k] += U * count_set_bits(doubly_occupied_sites)

                    # Chemical potential (mu term)
                    H[k, k] -= mu * (n_up + n_down)

                    # Nearest-neighbor interaction (V term)
                    n_total = state_up | state_down # Bitmask of all occupied sites
                    v_term_val = 0
                    for site1, site2 in INTERACTION_PAIRS:
                        n1 = int(bool(n_total & (1 << site1)))
                        n2 = int(bool(n_total & (1 << site2)))
                        # The term in Hamiltonian is (n_i,up + n_i,down)(n_j,up + n_j,down)
                        # which is n_i * n_j
                        n_i = count_set_bits(state_up & (1 << site1)) + count_set_bits(state_down & (1 << site1))
                        n_j = count_set_bits(state_up & (1 << site2)) + count_set_bits(state_down & (1 << site2))
                        v_term_val += n_i * n_j
                    H[k, k] -= V * v_term_val # The problem statement has V, not -V

            
                    # --- Off-diagonal terms (Hopping) ---
                    for site1, site2 in HOPPING_PAIRS:
                        # Hopping for spin-up
                        # Check if hopping is possible: site1 occupied, site2 empty OR site2 occupied, site1 empty
                        if bool(state_up & (1 << site1)) ^ bool(state_up & (1 << site2)):
                            # Hop from site1 to site2
                            if bool(state_up & (1 << site1)):
                                j_hop, i_hop = site1, site2
                            # Hop from site2 to site1
                            else:
                                j_hop, i_hop = site2, site1
                            
                            new_state_up = state_up ^ (1 << i_hop) ^ (1 << j_hop)
                            sign = get_fermionic_sign(state_up, i_hop, j_hop)
                            
                            new_i = map_up[new_state_up]
                            new_k = new_i * dim_down + j
                            H[k, new_k] += -t * sign

                        # Hopping for spin-down
                        if bool(state_down & (1 << site1)) ^ bool(state_down & (1 << site2)):
                            # Hop from site1 to site2
                            if bool(state_down & (1 << site1)):
                                j_hop, i_hop = site1, site2
                            # Hop from site2 to site1
                            else:
                                j_hop, i_hop = site2, site1
                            
                            new_state_down = state_down ^ (1 << i_hop) ^ (1 << j_hop)
                            sign = get_fermionic_sign(state_down, i_hop, j_hop)

                            new_j = map_down[new_state_down]
                            new_k = i * dim_down + new_j
                            H[k, new_k] += -t * sign
            
            # Diagonalize the Hamiltonian
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            all_eigenvalues.extend(eigenvalues)

            # --- Store results for specific tasks ---
            # Task 1: Lowest 6 eigenvalues in the (N_up=2, N_down=4) subspace
            if n_up == 2 and n_down == 4:
                print(f"(1) Lowest 6 eigenvalues in (N_up=2, N_down=4) subspace:")
                print(f"   {eigenvalues[:6]}\n")

            # Check for the true ground state
            if eigenvalues[0] < ground_state_info['energy']:
                ground_state_info['energy'] = eigenvalues[0]
                ground_state_info['vector'] = eigenvectors[:, 0]
                ground_state_info['basis_up'] = basis_up
                ground_state_info['basis_down'] = basis_down
                ground_state_info['n_up'] = n_up
                ground_state_info['n_down'] = n_down

    # --- Final Output ---
    # Task 2: Lowest 20 eigenvalues of the whole system
    all_eigenvalues.sort()
    print(f"(2) Lowest 20 eigenvalues of the whole system:")
    print(f"   {np.array(all_eigenvalues[:20])}\n")

    # Task 3: Calculate <n_i,sigma> for the true ground state
    gs_vector = ground_state_info['vector']
    gs_basis_up = ground_state_info['basis_up']
    gs_basis_down = ground_state_info['basis_down']
    dim_down_gs = len(gs_basis_down)

    density_up = np.zeros(N_SITES)
    density_down = np.zeros(N_SITES)
    
    for i in range(len(gs_basis_up)):
        for j in range(len(gs_basis_down)):
            k = i * dim_down_gs + j
            state_up = gs_basis_up[i]
            state_down = gs_basis_down[j]
            coeff_sq = gs_vector[k]**2
            
            for site in range(N_SITES):
                if (state_up >> site) & 1:
                    density_up[site] += coeff_sq
                if (state_down >> site) & 1:
                    density_down[site] += coeff_sq
    
    print(f"(3) Expectation value <n_i,sigma> for each site and spin:")
    print(f"   Density_Up_Spin:   {density_up}")
    print(f"   Density_Down_Spin: {density_down}\n")
    
    print(f"Ground state found in (N_up={ground_state_info['n_up']}, N_down={ground_state_info['n_down']}) subspace with energy E_0 = {ground_state_info['energy']}")
    print("=" * 40)


if __name__ == "__main__":
    main()