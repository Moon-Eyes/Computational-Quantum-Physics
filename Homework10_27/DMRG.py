#!/usr/bin/env python3
"""
finite DMRG for H = - sum (sigma_x sigma_x + g sigma_y sigma_y)
Compute ground state energy and entanglement entropy S(L).
Save and plot S(L). Fit S(L) for m=20, g=1.0 to extract central charge c.

This script is adapted from simple_dmrg_02_finite_system.py with modifications:
 - model replaced by XX (xx-yy with coefficient g) model: H_ij = -(sx sx + g sy sy)
 - during single_dmrg_step, compute and record entanglement entropy S(L)
 - run parameter scans required by the homework
"""

import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from math import sin, pi, log
import os

# ----- Model: Pauli matrices -----
sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype='d')
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype='complex128')
id1 = np.eye(2, dtype='d')

# We'll keep operators as dense numpy arrays for simplicity (small tutorial sizes).
# But use sparse kron for larger sizes when building superblock.
# Note: use complex dtype for matrices because sy is complex.
def H2(sx1, sy1, sx2, sy2, g):
    # two-site coupling operator for the bond joining two blocks/sites
    # H_bond = -( sx1 ⊗ sx2 + g * sy1 ⊗ sy2 )
    return -(np.kron(sx1, sx2) + g * np.kron(sy1, sy2))

# ----- Data structures -----
from collections import namedtuple
Block = namedtuple("Block", ["length", "basis_size", "operator_dict"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict"])

def is_valid_block(block):
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True

# initial single-site block
initial_block_template = Block(length=1, basis_size=2, operator_dict={
    "H": np.zeros((2,2), dtype='complex128'),
    "conn_sx": sx.astype('complex128'),
    "conn_sy": sy.astype('complex128'),
})

# enlarge block by one site
def enlarge_block(block, g):
    mblock = block.basis_size
    o = block.operator_dict
    # Kron products: new basis = block ⊗ site
    H_new = np.kron(o["H"], np.eye(2, dtype='complex128')) + np.kron(np.eye(mblock, dtype='complex128'), np.zeros((2,2),dtype='complex128')) \
            + H2(o["conn_sx"], o["conn_sy"], sx.astype('complex128'), sy.astype('complex128'), g)
    conn_sx_new = np.kron(np.eye(mblock, dtype='complex128'), sx.astype('complex128'))
    conn_sy_new = np.kron(np.eye(mblock, dtype='complex128'), sy.astype('complex128'))
    opdict = {"H": H_new, "conn_sx": conn_sx_new, "conn_sy": conn_sy_new}
    return EnlargedBlock(length=block.length+1, basis_size=mblock*2, operator_dict=opdict)

def rotate_and_truncate(operator, transformation_matrix):
    # operator (dense) rotated to new truncated basis: T^† O T
    # transformation_matrix: shape (old_dim, new_dim)
    return transformation_matrix.conj().T.dot(operator.dot(transformation_matrix))

def single_dmrg_step(sys, env, m, g):
    """
    Perform single DMRG step for system sys and environment env with parameter g.
    Returns: new_sys_block (truncated), energy (superblock ground energy),
             entropy (for the enlarged system size = sys_enl.length), and the evals of rho (for debugging)
    """
    assert is_valid_block(sys)
    assert is_valid_block(env)

    sys_enl = enlarge_block(sys, g)
    if sys is env:
        env_enl = sys_enl
    else:
        env_enl = enlarge_block(env, g)

    assert is_valid_block(sys_enl)
    assert is_valid_block(env_enl)

    # Build superblock Hamiltonian as sparse Kronecker to allow eigsh
    H_sys = csr_matrix(sys_enl.operator_dict["H"])
    H_env = csr_matrix(env_enl.operator_dict["H"])
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    I_sys = identity(m_sys_enl, format='csr', dtype='complex128')
    I_env = identity(m_env_enl, format='csr', dtype='complex128')

    # bond operator between sys_enl right edge and env_enl left edge
    # get conn operators
    conn_sx_sys = csr_matrix(sys_enl.operator_dict["conn_sx"])
    conn_sy_sys = csr_matrix(sys_enl.operator_dict["conn_sy"])
    conn_sx_env = csr_matrix(env_enl.operator_dict["conn_sx"])
    conn_sy_env = csr_matrix(env_enl.operator_dict["conn_sy"])
    # build H2 term as sparse using kron
    Hbond = kron(conn_sx_sys, conn_sx_env, format='csr') + g * kron(conn_sy_sys, conn_sy_env, format='csr')
    Hbond = -Hbond  # include minus sign

    superH = kron(H_sys, I_env, format='csr') + kron(I_sys, H_env, format='csr') + Hbond

    # Ensure hermiticity (small sym)
    superH = (superH + superH.getH()) * 0.5

    # find ground state with ARPACK (smallest algebraic)
    # for complex Hermitian, eigsh works; use which='SA' (smallest algebraic)
    try:
        evals_small, evecs = eigsh(superH, k=1, which='SA', return_eigenvectors=True)
    except Exception as e:
        # fallback: convert to dense (only for very small dims)
        print("eigsh failed, converting to dense (dims = {},{}). Exception: {}".format(m_sys_enl, m_env_enl, e))
        superH_dense = superH.toarray()
        evals_full, evecs_full = np.linalg.eigh(superH_dense)
        e0 = evals_full[0]
        psi0 = evecs_full[:,0]
        evecs = psi0.reshape((-1,1))
        evals_small = np.array([e0])

    energy = float(evals_small[0])
    psi0 = evecs[:,0]  # shape (m_sys_enl*m_env_enl,)

    # Build psi as matrix of shape (sys_enl.basis, env_enl.basis).
    # Due to kron ordering used, psi0 is in column-major order for env index varying fastest if constructed via kron(sys_enl, env_enl).
    # We reshape with order='F' to make psi matrix where rows correspond to system index.
    psi_mat = psi0.reshape((sys_enl.basis_size, env_enl.basis_size), order='F')

    # reduced density matrix for system: rho = psi_mat @ psi_mat^†
    rho = psi_mat.dot(psi_mat.conj().T)

    # diagonalize rho
    evals_rho, evecs_rho = np.linalg.eigh(rho)  # evals in ascending order
    # sort descending
    idx = np.argsort(evals_rho)[::-1]
    evals_rho = evals_rho[idx]
    evecs_rho = evecs_rho[:, idx]

    # compute entanglement entropy S = -sum p ln p (natural log)
    eps = 1e-12
    probs = np.real_if_close(evals_rho)
    probs[probs < 0] = 0.0
    probs = probs / np.sum(probs)  # normalize (just in case of numeric)
    mask = probs > eps
    S = -np.sum(probs[mask] * np.log(probs[mask]))

    # Build transformation matrix from the top-m eigenvectors (the columns of evecs_rho)
    my_m = min(len(probs), m)
    transformation_matrix = evecs_rho[:, :my_m]  # shape (sys_enl.basis_size, my_m)

    # rotate and truncate operators for the new truncated block
    new_ops = {}
    for name, op in sys_enl.operator_dict.items():
        new_ops[name] = rotate_and_truncate(op, transformation_matrix)

    newblock = Block(length=sys_enl.length, basis_size=my_m, operator_dict=new_ops)

    # return new block, energy (superblock), entropy for L = sys_enl.length, and full evals_rho
    return newblock, energy, S, evals_rho

def graphic(sys_block, env_block, sys_label="l"):
    assert sys_label in ("l","r")
    gstr = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    if sys_label == "r":
        gstr = gstr[::-1]
    return gstr

def finite_system_algorithm(N, m_warmup, m_sweep_list, g, verbose=False):
    """
    Run finite-system DMRG until final sweeps specified in m_sweep_list.
    Collect entropies S(L) during sweeps and ground state energies when superblock covers full N.
    Returns:
      - energies: list of energies observed when superblock size == N during sweeps
      - entropies_dict: mapping L -> last observed S(L) (float)
      - S_history: list of tuples (L, S) recorded over time (for more detail)
    """
    assert N % 2 == 0
    block_disk = {}  # store left and right blocks built during infinite stage

    # initialize
    block = Block(length=1, basis_size=2, operator_dict={
        "H": np.zeros((2,2), dtype='complex128'),
        "conn_sx": sx.astype('complex128'),
        "conn_sy": sy.astype('complex128'),
    })
    block_disk[("l", block.length)] = block
    block_disk[("r", block.length)] = block

    # infinite algorithm warmup to build up to L = N/2 (mirror)
    while 2 * block.length < N:
        if verbose:
            print("Building (infinite) L=", block.length*2+2)
        block, energy_dummy, S_dummy, _ = single_dmrg_step(block, block, m_warmup, g)
        block_disk[("l", block.length)] = block
        block_disk[("r", block.length)] = block
        if verbose:
            print("E/L (warmup) = ", energy_dummy / (block.length * 2))

    # Now finite sweeps
    sys_label, env_label = "l", "r"
    sys_block = block
    energies = []
    entropies_dict = {}  # L -> last S(L)
    S_history = []

    for m in m_sweep_list:
        if verbose:
            print("Starting sweep with m =", m)
        while True:
            env_block = block_disk[(env_label, N - sys_block.length - 2)]
            # if env_block.length==1 we've reached the end -> reverse direction
            if env_block.length == 1:
                sys_block, env_block = env_block, sys_block
                sys_label, env_label = env_label, sys_label

            if verbose:
                print(graphic(sys_block, env_block, sys_label))
            sys_block, energy, S, evals_rho = single_dmrg_step(sys_block, env_block, m, g)

            # record entropy at L = sys_block.length (note: single_dmrg_step returned newblock with length = sys_enl.length)
            L_here = sys_block.length
            entropies_dict[L_here] = S
            S_history.append((L_here, S))

            # If the current superblock covers full N? In this code energy corresponds to the superblock energy just computed.
            # The superblock size is (sys_enl.length + env_enl.length) which during finite sweep should equal N at the appropriate steps.
            # Simple reliable test: if sys_block.length + (N - sys_block.length - 2) + 2 == N, i.e., always true; so instead record energies when sys_block.length*2 == N
            # We'll record energies when the two blocks meet in the middle: i.e., when sys_label == 'l' and 2*sys_block.length == N
            if sys_label == "l" and 2 * L_here == N:
                energies.append(energy)
                if verbose:
                    print("Recorded energy for full chain: E = {:.12f}".format(energy))

            block_disk[(sys_label, sys_block.length)] = sys_block

            # check end of sweep condition
            if sys_label == "l" and 2 * sys_block.length == N:
                # completed a full sweep left->right->left
                break

    # For entropies for all L from 1..N-1, fill missing L with NaN
    S_array = np.empty(N+1)
    S_array[:] = np.nan
    for L in range(1, N):
        if L in entropies_dict:
            S_array[L] = entropies_dict[L]
    # return energies list and entropies array (index L)
    return energies, S_array, S_history

# ---------- Utility plotting and fitting ----------
def plot_S_vs_L(S_array, N, title, fname):
    Ls = np.arange(1, N)
    Svals = S_array[1:N]
    plt.figure(figsize=(8,5))
    plt.plot(Ls, Svals, marker='o', linestyle='-')
    plt.xlabel("L")
    plt.ylabel("S(L)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def fit_c_from_S(S_array, N, Lmin=2, Lmax=None):
    if Lmax is None:
        Lmax = N-2
    Ls = np.arange(Lmin, Lmax+1)
    x = np.log((N/np.pi) * np.sin(np.pi * Ls / N))
    y = S_array[Ls]
    # remove NaN or -inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 3:
        raise RuntimeError("Not enough points to fit")
    # linear fit y = a * x + b
    a, b = np.polyfit(x, y, 1)
    c_est = 6.0 * a
    return c_est, a, b, x, y

# ---------- Main runner for homework parameter sets ----------
def run_homework(N=40, output_dir="dmrg_results", do_plots=True, verbose=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = {}

    # Task (1): m=10, g in [0.5, 1.0, 1.5]
    m_fixed = 10
    gs = [0.5, 1.0, 1.5]
    for g in gs:
        print("Running N={}, m={}, g={}".format(N, m_fixed, g))
        energies, S_array, S_hist = finite_system_algorithm(N=N, m_warmup=m_fixed, m_sweep_list=[m_fixed, m_fixed*2], g=g, verbose=verbose)
        # pick last recorded energy as ground state energy
        E_gs = energies[-1] if len(energies) > 0 else None
        print("g={}, energies observed (last) = {}".format(g, E_gs))
        results[("m", m_fixed, "g", g)] = {"energies": energies, "S_array": S_array, "S_history": S_hist}
        if do_plots:
            plot_S_vs_L(S_array, N, title=f"S(L), N={N}, m={m_fixed}, g={g}", fname=os.path.join(output_dir, f"S_L_N{N}_m{m_fixed}_g{g}.png"))
            # save S array
            np.savetxt(os.path.join(output_dir, f"S_L_N{N}_m{m_fixed}_g{g}.csv"), S_array, header="index L from 0..N (L=0 and L=N unused)")

    # Task (2): g=1.0, m in [10,20,30]
    g_fixed = 1.0
    ms = [10, 20, 30]
    for m in ms:
        print("Running N={}, m={}, g={}".format(N, m, g_fixed))
        energies, S_array, S_hist = finite_system_algorithm(N=N, m_warmup=m, m_sweep_list=[m, m*2], g=g_fixed, verbose=verbose)
        E_gs = energies[-1] if len(energies) > 0 else None
        print("m={}, energies observed (last) = {}".format(m, E_gs))
        results[("m", m, "g", g_fixed)] = {"energies": energies, "S_array": S_array, "S_history": S_hist}
        if do_plots:
            plot_S_vs_L(S_array, N, title=f"S(L), N={N}, m={m}, g={g_fixed}", fname=os.path.join(output_dir, f"S_L_N{N}_m{m}_g{g_fixed}.png"))
            np.savetxt(os.path.join(output_dir, f"S_L_N{N}_m{m}_g{g_fixed}.csv"), S_array, header="index L from 0..N (L=0 and L=N unused)")

    # Task (3): for m=20, g=1.0 fit S(L) to extract c
    key = ("m", 20, "g", 1.0)
    if key in results:
        S_array = results[key]["S_array"]
        # choose range for fitting: avoid very small L and L very close to N
        try:
            c_est, a, b, xvals, yvals = fit_c_from_S(S_array, N, Lmin=3, Lmax=N-3)
            print("Fitted c (m=20,g=1.0): c = {:.6f} (slope a={:.6f}, intercept b={:.6f})".format(c_est, a, b))
            # plot fit
            plt.figure(figsize=(8,5))
            plt.scatter(xvals, yvals, label='data')
            plt.plot(xvals, a*xvals + b, 'r-', label=f'fit: slope={a:.4f}')
            plt.xlabel("ln((N/pi) sin(pi L / N))")
            plt.ylabel("S(L)")
            plt.legend()
            plt.title(f"Fit for m=20,g=1.0: c={c_est:.4f}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"fit_S_vs_logterm_N{N}_m20_g1.0.png"), dpi=200)
            plt.close()
            results[("fit", key)] = {"c": c_est, "slope": a, "intercept": b}
        except Exception as e:
            print("Fitting failed:", e)

    # Save summary
    np.savez(os.path.join(output_dir, "dmrg_results_summary.npz"), **{
        f"{k}": v["S_array"] if isinstance(v, dict) and "S_array" in v else v
        for k, v in results.items()
    })
    print("All done. Results saved in", output_dir)
    return results

if __name__ == "__main__":
    # run with defaults
    results = run_homework(N=40, output_dir="dmrg_results", do_plots=True, verbose=False)
