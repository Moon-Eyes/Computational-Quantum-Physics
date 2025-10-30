from __future__ import print_function, division  # requires Python >= 2.6

# numpy 和 scipy 导入
import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK

# 新增导入，用于绘图和拟合
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 我们将使用 python 的 "namedtuple" 来表示 Block 和 EnlargedBlock
from collections import namedtuple

Block = namedtuple("Block", ["length", "basis_size", "operator_dict"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict"])

def is_valid_block(block):
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True

is_valid_enlarged_block = is_valid_block

#
# --- 作业模型特定代码 (XX 链) ---
#
model_d = 2  # 单个节点的基矢大小

# S^+ 和 S^- 算符
Sp1 = np.array([[0, 1], [0, 0]], dtype='d')  # single-site S^+
Sm1 = np.array([[0, 0], [1, 0]], dtype='d')  # single-site S^-

# 单个节点的哈密顿量为零
H1 = np.array([[0, 0], [0, 0]], dtype='d')

def H2(Sp1, Sm1, Sp2, Sm2, g):
    """
    给定两个节点（或块）上的 S+ 和 S- 算符，
    返回连接这两个节点的哈密顿量项。
    H_interaction = - (sigma_x_1 * sigma_x_2 + g * sigma_y_1 * sigma_y_2)
    
    其中:
    sigma_x = S+ + S-
    sigma_y = -i * (S+ - S-)
    
    sigma_x_1 * sigma_x_2 = (Sp1 + Sm1) kron (Sp2 + Sm2)
    sigma_y_1 * sigma_y_2 = (-i)(Sp1 - Sm1) kron (-i)(Sp2 - Sm2)
                         = -1 * (Sp1 - Sm1) kron (Sp2 - Sm2)
                         
    H_interaction = -[ (Sp1 + Sm1) kron (Sp2 + Sm2) - g * (Sp1 - Sm1) kron (Sp2 - Sm2) ]
    """
    # 注意: 原始代码使用 S^z 和 S^+ 
    # 我们修改为使用 S^+ 和 S^-
    
    # sigma_x * sigma_x 项
    H_xx = kron(Sp1 + Sm1, Sp2 + Sm2)
    
    # sigma_y * sigma_y 项
    H_yy = -kron(Sp1 - Sm1, Sp2 - Sm2)
    
    return -(H_xx + g * H_yy)

# 初始块现在存储 S+ 和 S-
initial_block = Block(length=1, basis_size=model_d, operator_dict={
    "H": H1,
    "conn_Sp": Sp1, # conn_Sp 替代 conn_Sz 
    "conn_Sm": Sm1, # 新增 conn_Sm
})

def enlarge_block(block, g):
    """
    此函数将提供的 Block 扩大一个节点，返回一个 EnlargedBlock。
    需要传递 'g' 来构建 H2 项。
    """
    mblock = block.basis_size
    o = block.operator_dict

    # 新节点的算符
    Sp1_site = np.array([[0, 1], [0, 0]], dtype='d')
    Sm1_site = np.array([[0, 0], [1, 0]], dtype='d')
    H1_site = np.array([[0, 0], [0, 0]], dtype='d')

    # 为扩大后的块创建新算符
    enlarged_operator_dict = {
        "H": kron(o["H"], identity(model_d)) + kron(identity(mblock), H1_site) + \
             H2(o["conn_Sp"], o["conn_Sm"], Sp1_site, Sm1_site, g), # 修改 H2 调用
        "conn_Sp": kron(identity(mblock), Sp1_site),
        "conn_Sm": kron(identity(mblock), Sm1_site), # 传播 conn_Sm
    }

    return EnlargedBlock(length=(block.length + 1),
                         basis_size=(block.basis_size * model_d),
                         operator_dict=enlarged_operator_dict)

def rotate_and_truncate(operator, transformation_matrix):
    """
    将算符变换到由 `transformation_matrix` 给出的新的（可能被截断的）基上。
    
    """
    return transformation_matrix.conjugate().transpose().dot(operator.dot(transformation_matrix))

def single_dmrg_step(sys, env, m, g):
    """
    执行单步DMRG，使用 `sys` 作为系统，`env` 作为环境，
    在新基中最多保留 `m` 个状态。
    
    修改：传递 'g' 并返回能量和纠缠熵。
    """
    assert is_valid_block(sys)
    assert is_valid_block(env)

    # 扩大每个块
    sys_enl = enlarge_block(sys, g) # 传递 g
    if sys is env:
        env_enl = sys_enl
    else:
        env_enl = enlarge_block(env, g) # 传递 g

    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)

    # 构建完整的超块哈密顿量
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict
    
    superblock_hamiltonian = kron(sys_enl_op["H"], identity(m_env_enl)) + \
                             kron(identity(m_sys_enl), env_enl_op["H"]) + \
                             H2(sys_enl_op["conn_Sp"], sys_enl_op["conn_Sm"], 
                                env_enl_op["conn_Sp"], env_enl_op["conn_Sm"], g) # 修改 H2 调用

    # 寻找超块基态
    (energy,), psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")

    # 构建系统的约化密度矩阵
    psi0 = psi0.reshape([sys_enl.basis_size, -1], order="C")
    rho = np.dot(psi0, psi0.conjugate().transpose())

    # 对角化约化密度矩阵
    evals, evecs = np.linalg.eigh(rho)

    # --- 新增：计算纠缠熵 ---
    # S = -Tr(rho * log(rho)) = -sum(lambda * log(lambda))
    # 作业要求 S(L) = -Tr[rho_L * ln(rho_L)]
    entropy = 0.0
    for val in evals:
        if val > 1e-12:  # 避免 log(0)
            entropy -= val * np.log(val) # np.log 是 ln
    # --------------------------

    # 根据本征值大小对本征矢排序
    possible_eigenstates = []
    for eval, evec in zip(evals, evecs.transpose()):
        possible_eigenstates.append((eval, evec))
    # 
    possible_eigenstates.sort(reverse=True, key=lambda x: x[0])

    # 构建变换矩阵
    my_m = min(len(possible_eigenstates), m)
    transformation_matrix = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
        transformation_matrix[:, i] = evec

    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    print(f"L_sys={sys_enl.length}, L_env={env_enl.length}, m={my_m}, trunc error: {truncation_error:.2e}")

    # 旋转并截断每个算符
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict)

    return newblock, energy, entropy # 返回 entropy

def graphic(sys_block, env_block, sys_label="l"):
    """
    返回DMRG步骤的图形表示。 
    """
    assert sys_label in ("l", "r")
    graphic = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    if sys_label == "r":
        graphic = graphic[::-1]
    return graphic

def finite_system_algorithm_homework(L, m, g):
    """
    为作业特定需求修改的有限系统算法。
    - 使用 'g' 参数
    - 运行固定的扫描次数
    - 在最后一次扫描中收集纠缠熵
    """
    assert L % 2 == 0

    block_disk = {}  # 块对象的“磁盘”存储 

    # --- 预热 (Warmup) ---
    # 使用无限系统算法构建到 L/2 - 1
    print("--- 预热 (Infinite System Algorithm) ---")
    block = initial_block
    block_disk["l", block.length] = block
    block_disk["r", block.length] = block
    while 2 * block.length < L:
        print(graphic(block, block))
        # single_dmrg_step 现在需要 g，并返回 3 个值
        block, energy, _ = single_dmrg_step(block, block, m=m, g=g)
        print(f"E/L = {energy / (block.length * 2):.10f}")
        block_disk["l", block.length] = block
        block_disk["r", block.length] = block
    
    # --- 扫描 (Sweeps) ---
    # 执行 5 次扫描。作业只指定了 m，我们就用这个 m 进行扫描。
    m_sweep_list = [m] * 5
    
    sys_label, env_label = "l", "r"
    sys_block = block; del block  # 重命名变量 
    
    entropies = {} # 存储 L -> S(L)
    final_energy = 0.0

    for sweep_num, m_sweep in enumerate(m_sweep_list):
        print(f"\n--- 开始扫描 {sweep_num+1}/{len(m_sweep_list)} (m={m_sweep}) ---")
        
        # 完整的 L->R 和 R->L 扫描
        while True:
            # 从“磁盘”加载环境块 
            env_block = block_disk[env_label, L - sys_block.length - 2]
            
            # 检查是否到达链的末端
            if env_block.length == 1:
                # 
                # 这是在转向之前的最后一步
                print(graphic(sys_block, env_block, sys_label))
                sys_block, energy, entropy = single_dmrg_step(sys_block, env_block, m=m_sweep, g=g)
                final_energy = energy
                print(f"E/L = {energy / L:.10f}")

                # 存储熵
                L_left = sys_block.length if sys_label == "l" else L - sys_block.length
                if sweep_num == len(m_sweep_list) - 1: # 仅在最后一次扫描时
                    entropies[L_left] = entropy
                
                block_disk[sys_label, sys_block.length] = sys_block

                # 转向 
                sys_label, env_label = env_label, sys_label
                sys_block, env_block = env_block, sys_block
                # 继续循环，执行转向后的第一步
            
            # 执行DMRG步骤
            print(graphic(sys_block, env_block, sys_label))
            sys_block, energy, entropy = single_dmrg_step(sys_block, env_block, m=m_sweep, g=g)
            final_energy = energy
            print(f"E/L = {energy / L:.10f}")

            # 存储熵
            L_left = sys_block.length if sys_label == "l" else L - sys_block.length
            if sweep_num == len(m_sweep_list) - 1: # 仅在最后一次扫描时
                entropies[L_left] = entropy

            # 保存块 
            block_disk[sys_label, sys_block.length] = sys_block

            # 检查是否完成了一次完整的扫描 
            if sys_label == "l" and 2 * sys_block.length == L:
                break  # 从 "while True" 循环中退出

    # 排序熵数据
    L_values = sorted(entropies.keys())
    S_values = [entropies[l] for l in L_values]
    
    return final_energy, L_values, S_values

#
# --- 作业 (3): 拟合函数 ---
#
def fit_function(x, c, c_prime):
    """
    S(L) = (c / 6) * x + c_prime
    其中 x = log( (N/pi) * sin(pi*L/N) )
    """
    return (c / 6) * x + c_prime

#
# --- 主执行函数 ---
#
def main():
    """
    执行作业中要求的所有任务。
    """
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=200)

    N = 40 # 作业要求
    
    # 定义要运行的参数
    # 任务 (1)
    params_m10 = [(10, 0.5), (10, 1.0), (10, 1.5)]
    # 任务 (2)
    params_g1 = [(20, 1.0), (30, 1.0)] # (10, 1.0) 已经在上面
    
    params_to_run = params_m10 + params_g1
    all_results = {}

    print("=== 开始 DMRG 计算 ===")
    
    # --- 任务 (1) 和 (2): 运行DMRG并收集数据 ---
    
    # 准备 S(L) vs L 的图
    plt.figure(figsize=(10, 7))
    ax1 = plt.gca()
    
    for m, g in params_to_run:
        print(f"\n--- 正在运行 (N, m, g) = ({N}, {m}, {g}) ---")
        energy, L_vals, S_vals = finite_system_algorithm_homework(L=N, m=m, g=g)
        all_results[(m, g)] = (energy, L_vals, S_vals)
        
        # 任务 (1): 打印数据
        print("\n--- 结果 (任务 1) ---")
        print(f"(N, m, g) = ({N}, {m}, {g})")
        print(f"Ground State Energy: {energy:.10f}")
        L_arr_str = np.array2string(np.array(L_vals), precision=5, max_line_width=120)
        S_arr_str = np.array2string(np.array(S_vals), precision=5, max_line_width=120)
        print(f"L: {L_arr_str}")
        print(f"EE: {S_arr_str}")
        print("-----------------------\n")
        
        # 任务 (2): 绘制数据
        label = f"m={m}, g={g}"
        # 模仿 output.png 的样式
        style = '--' if m == 10 else '-'
        marker = 'o' if g == 1.0 else ('^' if g == 0.5 else 's')
        ax1.plot(L_vals, S_vals, marker=marker, linestyle=style, label=label, markersize=4)

    # 完成 S(L) vs L 的图
    ax1.set_xlabel("L (系统块长度)")
    ax1.set_ylabel("纠缠熵 S(L)")
    ax1.set_title(f"纠缠熵 vs. 系统块长度 (N={N})")
    ax1.legend()
    ax1.grid(True)
    plt.savefig("task_2_S_vs_L.png")
    print("\n[已保存 S(L) vs L 图像至 'task_2_S_vs_L.png']")

    # --- 任务 (3): 拟合 m=20, g=1.0 ---
    print("\n--- 任务 (3): 拟合 (N, m, g) = (40, 20, 1.0) ---")
    
    m_fit, g_fit = 20, 1.0
    try:
        energy_fit, L_fit, S_fit = all_results[(m_fit, g_fit)]
    except KeyError:
        print(f"错误: 没有 (m={m_fit}, g={g_fit}) 的结果，正在重新运行...")
        energy_fit, L_fit, S_fit = finite_system_algorithm_homework(L=N, m=m_fit, g=g_fit)
        
    L_arr = np.array(L_fit)
    S_arr = np.array(S_fit)
    
    # 准备拟合数据
    # x = log( (N/pi) * sin(pi*L/N) )
    x_data = np.log((N / np.pi) * np.sin(np.pi * L_arr / N))
    y_data = S_arr
    
    # 执行拟合
    popt, pcov = curve_fit(fit_function, x_data, y_data)
    
    c_fit = popt[0]
    c_prime_fit = popt[1]
    
    print(f"(3): central_charge = {c_fit:.10f}")
    print(f"    intercept = {c_prime_fit:.10f}")
    
    # 绘制 S(L) vs L (单独)
    plt.figure(figsize=(10, 7))
    plt.plot(L_arr, S_arr, 'bo-', label=f'DMRG 数据 (m={m_fit}, g={g_fit})')
    # 绘制拟合曲线
    S_fit_curve = fit_function(x_data, c_fit, c_prime_fit)
    plt.plot(L_arr, S_fit_curve, 'r--', label=f'拟合 (c={c_fit:.4f})')
    plt.xlabel("L (系统块长度)")
    plt.ylabel("纠缠熵 S(L)")
    plt.title(f"S(L) vs. L (m={m_fit}, g={g_fit})")
    plt.legend()
    plt.grid(True)
    plt.savefig("task_3_S_vs_L_fit.png")
    print("[已保存 S(L) vs L (拟合) 图像至 'task_3_S_vs_L_fit.png']")

    # 绘制 S(L) vs x_data (线性拟合图)
    plt.figure(figsize=(10, 7))
    plt.plot(x_data, y_data, 'bo', label='DMRG 数据')
    plt.plot(x_data, fit_function(x_data, *popt), 'r-', label=f'线性拟合: c/6 * x + c\'\n c = {c_fit:.4f}')
    plt.xlabel("x = log( (N/pi) * sin(pi*L/N) )")
    plt.ylabel("纠缠熵 S(L)")
    plt.title(f"中心电荷拟合 (m={m_fit}, g={g_fit})")
    plt.legend()
    plt.grid(True)
    plt.savefig("task_3_central_charge_fit.png")
    print("[已保存中心电荷拟合图像至 'task_3_central_charge_fit.png']")

    print("\n=== 所有任务完成 ===")
    plt.show() # 显示所有图像

if __name__ == "__main__":
    main()