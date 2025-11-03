from __future__ import print_function, division
import numpy as np
from scipy.sparse import kron, identity, csr_matrix, isspmatrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from collections import namedtuple
Block = namedtuple("Block", ["length", "basis_size", "operator_dict"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict"])

def is_valid_block(block):
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True
is_valid_enlarged_block = is_valid_block

model_d = 2

Sp1 = np.array([[0, 1], [0, 0]], dtype='d')
Sm1 = np.array([[0, 0], [1, 0]], dtype='d')
H1 = np.array([[0, 0], [0, 0]], dtype='d')

initial_block = Block(length=1, basis_size=model_d, operator_dict={
    "H": H1,
    "conn_Sp": Sp1,
    "conn_Sm": Sm1,
})

def H2(Sp1, Sm1, Sp2, Sm2, g):
    # Ensure inputs can be sparse or dense; use scipy.sparse.kron for consistency
    # But return dense or sparse consistently to caller.
    A = kron(Sp1 + Sm1, Sp2 + Sm2)
    B = -kron(Sp1 - Sm1, Sp2 - Sm2)
    return -(A + g * B)

def enlarge_block(block, g):
    mblock = block.basis_size
    o = block.operator_dict

    # New site operators (dense)
    Sp1_site = np.array([[0, 1], [0, 0]], dtype='d')
    Sm1_site = np.array([[0, 0], [1, 0]], dtype='d')
    H1_site = np.array([[0, 0], [0, 0]], dtype='d')

    # Build enlarged operators. Use sparse identity for kron with existing operators.
    # Ensure o["H"] might be sparse or dense; convert to sparse when needed.
    left_H = kron(o["H"], identity(model_d))
    right_H = kron(identity(mblock), H1_site)
    conn_term = H2(o["conn_Sp"], o["conn_Sm"], Sp1_site, Sm1_site, g)

    H_enl = left_H + right_H + conn_term

    # conn operators of enlarged block (dense ndarray)
    conn_Sp_enl = kron(identity(mblock), Sp1_site)
    conn_Sm_enl = kron(identity(mblock), Sm1_site)

    # Convert all to csr sparse for consistency (H may be sparse)
    if not isspmatrix(H_enl):
        H_enl = csr_matrix(H_enl)
    else:
        H_enl = H_enl.tocsr()
    if not isspmatrix(conn_Sp_enl):
        conn_Sp_enl = csr_matrix(conn_Sp_enl)
    else:
        conn_Sp_enl = conn_Sp_enl.tocsr()
    if not isspmatrix(conn_Sm_enl):
        conn_Sm_enl = csr_matrix(conn_Sm_enl)
    else:
        conn_Sm_enl = conn_Sm_enl.tocsr()

    enlarged_operator_dict = {
        "H": H_enl,
        "conn_Sp": conn_Sp_enl,
        "conn_Sm": conn_Sm_enl,
    }

    return EnlargedBlock(length=(block.length + 1),
                         basis_size=(block.basis_size * model_d),
                         operator_dict=enlarged_operator_dict)


def rotate_and_truncate(operator, transformation_matrix):
    """
    Perform U^\dagger O U and return dense ndarray result.
    Ensure operator is ndarray (dense) before multiplication.
    transformation_matrix is ndarray with shape (old_dim, new_dim).
    """
    # If operator is sparse, convert to dense ndarray
    if isspmatrix(operator):
        op = operator.toarray()
    else:
        op = np.asarray(operator)

    # Ensure transformation matrix is complex dtype for safety
    U = np.asarray(transformation_matrix, dtype=complex)
    # U^\dagger O U
    return (U.conj().T).dot(op.dot(U))


def single_dmrg_step(sys, env, m, g):
    """
    Single DMRG step: enlarge blocks, build superblock H, find ground state,
    compute reduced density matrix for system, do truncation & return new block.
    """
    assert is_valid_block(sys)
    assert is_valid_block(env)

    # Enlarge
    sys_enl = enlarge_block(sys, g)
    if sys is env:
        env_enl = sys_enl
    else:
        env_enl = enlarge_block(env, g)

    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)

    # Build superblock Hamiltonian (sparse CSR)
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict

    H_sys = sys_enl_op["H"]
    H_env = env_enl_op["H"]
    # Ensure csr
    if not isspmatrix(H_sys):
        H_sys = csr_matrix(H_sys)
    else:
        H_sys = H_sys.tocsr()
    if not isspmatrix(H_env):
        H_env = csr_matrix(H_env)
    else:
        H_env = H_env.tocsr()

    superblock_hamiltonian = kron(H_sys, identity(m_env_enl)) + kron(identity(m_sys_enl), H_env) \
                             + H2(sys_enl_op["conn_Sp"], sys_enl_op["conn_Sm"],
                                  env_enl_op["conn_Sp"], env_enl_op["conn_Sm"], g)
    if not isspmatrix(superblock_hamiltonian):
        superblock_hamiltonian = csr_matrix(superblock_hamiltonian)
    else:
        superblock_hamiltonian = superblock_hamiltonian.tocsr()

    # Find ground state (k=1). eigsh returns (eigvals, eigvecs) where eigvecs columns are eigenvectors.
    eigvals, eigvecs = eigsh(superblock_hamiltonian, k=1, which="SA")
    energy = float(eigvals[0])
    psi = eigvecs[:, 0]  # explicit 1D vector

    # Reshape psi into matrix of shape (m_sys_enl, m_env_enl) with C-ordering
    psi_mat = psi.reshape((m_sys_enl, m_env_enl), order='C')

    # Reduced density matrix for system: rho_sys = psi_mat @ psi_mat^\dagger
    rho = np.dot(psi_mat, psi_mat.conj().T)
    # Ensure hermitian numeric symmetry
    rho = (rho + rho.conj().T) / 2.0

    # Diagonalize rho (dense)
    evals, evecs = np.linalg.eigh(rho)
    # sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Entanglement entropy
    entropy = 0.0
    for val in evals:
        if val > 1e-12:
            entropy -= val * np.log(val)

    # Truncation: keep the largest m eigenstates
    my_m = min(len(evals), m)
    transformation_matrix = np.asarray(evecs[:, :my_m], dtype=complex, order='F')

    # Rotate and truncate operators (convert to dense inside)
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

    # Convert new_operator_dict entries to dense ndarray (and ensure shapes match my_m)
    for k in list(new_operator_dict.keys()):
        new_operator_dict[k] = np.asarray(new_operator_dict[k], dtype=complex)

    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict)

    return newblock, energy, entropy


def graphic(sys_block, env_block, sys_label="l"):

    assert sys_label in ("l", "r")
    graphic = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    if sys_label == "r":
        graphic = graphic[::-1]
    return graphic


def finite_system_algorithm_homework(L, m, g):
    """
    Finite system algorithm modified for this task.
    Run a fixed number of scans, and collect entanglement entropy in the last scan.
    """
    assert L % 2 == 0
    block_disk = {}

    # --- Warmup ---
    block = initial_block
    block_disk["l", block.length] = block
    block_disk["r", block.length] = block
    while 2 * block.length < L:
        block, energy, _ = single_dmrg_step(block, block, m=m, g=g)
        block_disk["l", block.length] = block
        block_disk["r", block.length] = block

    # --- Sweep ---
    m_sweep_list = [m] * 5

    sys_label, env_label = "l", "r"
    sys_block = block; del block
    entropies = {}
    final_energy = 0.0

    for sweep_num, m_sweep in enumerate(m_sweep_list):
        while True:
            # pick env block that complements sys_block to fill L (two central sites counted in single_dmrg_step)
            # NOTE: index formula kept as original but be careful of keys
            env_block = block_disk[env_label, L - sys_block.length - 2]

            if env_block.length == 1:
                # special case: perform step and then turn back (but do not fall through to do another step this iteration)
                sys_block, energy, entropy = single_dmrg_step(sys_block, env_block, m=m_sweep, g=g)
                final_energy = energy

                L_left = sys_block.length if sys_label == "l" else L - sys_block.length
                if sweep_num == len(m_sweep_list) - 1:
                    entropies[L_left] = entropy

                block_disk[sys_label, sys_block.length] = sys_block
                # swap roles and continue next iteration
                sys_label, env_label = env_label, sys_label
                sys_block, env_block = env_block, sys_block
                continue  # IMPORTANT: avoid duplicate step in this iteration

            # Normal DMRG step
            sys_block, energy, entropy = single_dmrg_step(sys_block, env_block, m=m_sweep, g=g)
            final_energy = energy

            L_left = sys_block.length if sys_label == "l" else L - sys_block.length
            if sweep_num == len(m_sweep_list) - 1:
                entropies[L_left] = entropy

            block_disk[sys_label, sys_block.length] = sys_block

            # when we reach middle (left-moving sweeps)
            if sys_label == "l" and 2 * sys_block.length == L:
                break

    L_values = sorted(entropies.keys())
    S_values = [entropies[l] for l in L_values]

    return final_energy, L_values, S_values


def fit_function(x, c, c_prime):
    return (c / 6) * x + c_prime


def main():
    np.set_printoptions(precision=8, suppress=True, threshold=10000, linewidth=200)
    np.random.seed(0)

    N = 10
    params_to_run = [
        (10, 0.5),
        (10, 1.0),
        (10, 1.5),
        (10, 1.0),
        (20, 1.0),
        (30, 1.0)
    ]
    all_results = {}

    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    print("(1) and (2)")

    for m, g in params_to_run:
        energy, L_vals, S_vals = finite_system_algorithm_homework(L=N, m=m, g=g)
        all_results[(m, g)] = (energy, L_vals, S_vals)

        # Task 1
        print(f"(N, m, g)=({N}, {m}, {g})")
        print(f"Energy: {energy:.13f}")

        L_arr_str = np.array2string(np.array(L_vals), precision=0, separator=' ', max_line_width=120)
        print(f"L: {L_arr_str}")

        S_arr_str = np.array2string(np.array(S_vals), precision=8, suppress_small=True, max_line_width=120)
        print(f"EE: {S_arr_str}\n")

        # Task 2 plotting styles (keep same as original sample mapping)
        label = f"m={m}, g={g}"
        style, color, zorder = '-', 'black', 1  # default
        if m == 10 and g == 0.5: style, color, zorder = '-', 'green', 5
        elif m == 10 and g == 1.0: style, color, zorder = '-', 'blue', 5
        elif m == 10 and g == 1.5: style, color, zorder = '-', 'orange', 5
        elif m == 10 and g == 1.0: style, color, zorder = '--', 'red', 3
        elif m == 20 and g == 1.0: style, color, zorder = '--', 'purple', 3
        elif m == 30 and g == 1.0: style, color, zorder = ':', 'cyan', 1

        ax1.plot(L_vals, S_vals, marker='o', linestyle=style, color=color, label=label, markersize=4, zorder=zorder)

    ax1.set_xlabel("L")
    ax1.set_ylabel("Entanglement Entropy")
    ax1.legend()
    plt.savefig("sample_plot_S_vs_L.png")

    # Task (3) -- fit for m=20, g=1.0
    m_fit, g_fit = 20, 1.0
    N_fit = N
    try:
        energy_fit, L_fit, S_fit = all_results[(m_fit, g_fit)]
    except KeyError:
        energy_fit, L_fit, S_fit = finite_system_algorithm_homework(L=N_fit, m=m_fit, g=g_fit)

    L_arr = np.array(L_fit)
    S_arr = np.array(S_fit)

    x_data = np.log((N_fit / np.pi) * np.sin(np.pi * L_arr / N_fit))
    y_data = S_arr
    popt, pcov = curve_fit(fit_function, x_data, y_data)
    c_fit = popt[0]
    c_prime_fit = popt[1]

    y_fit_data = fit_function(x_data, *popt)
    r_value = np.corrcoef(y_data, y_fit_data)[0, 1]

    print(f"(3): central_charge= {c_fit} intercept= {c_prime_fit} r_value= {r_value}")

    plt.figure(figsize=(8, 6))
    plt.plot(L_arr, S_arr, 'o-')
    S_fit_curve = fit_function(x_data, c_fit, c_prime_fit)
    plt.plot(L_arr, S_fit_curve, '-')
    plt.xlabel("L")
    plt.ylabel("Entanglement Entropy")
    plt.savefig("sample_plot_S_vs_L_fit.png")

    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, 'o-')
    plt.plot(x_data, fit_function(x_data, *popt), '-')
    plt.xlabel("1/6 Log(N/pi*sin(pi*L/N))")
    plt.ylabel("Entanglement Entropy")
    plt.savefig("sample_plot_central_charge_fit.png")

    plt.show()


if __name__ == "__main__":
    main()
