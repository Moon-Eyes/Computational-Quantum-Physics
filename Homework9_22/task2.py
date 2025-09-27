import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

# Parameters setting

N = 101  
F = 0.1  
k0 = np.pi /2   
alpha = 0.15  
N0 = 51  
t_max = 100  
t_val = 42
dt = 0.1   
t_points = np.arange(0, t_max + dt, dt)
j_list = [10, 20, 30, 40, 50]  
 

# (1) Construct Hamiltonian and compute eigenvalues

diag = F * np.arange(1, N + 1)
off_diag = - np.ones(N - 1)
H_matrix = diags([off_diag, diag, off_diag], [-1, 0, 1]).toarray()

eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)


# (2) compute the probability density

j_sites = np.arange(1, N + 1)
psi0 = np.exp(- (alpha**2 / 2) * (j_sites - N0)**2) * np.exp(1j * k0 * j_sites)
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2))

c_n0 = np.matmul(eigenvectors.T.conj(), psi0)

psi_t = np.zeros((N, len(t_points)), dtype=complex)  
for i, t in enumerate(t_points):
    c_nt = c_n0 * np.exp(-1j * eigenvalues * t)
    psi_t[:, i] = np.matmul(eigenvectors, c_nt)

probability_density = np.abs(psi_t) ** 2

time_index = np.where(np.isclose(t_points, t_val))[0][0]
site_index = [j - 1 for j in j_list]
prob_values = probability_density[site_index, time_index]


# (3) Plot the probability density heatmap

plt.figure(figsize=(10, 8))
plt.imshow(probability_density, aspect='auto', origin='lower', extent=[0, t_max, 1, N], cmap='jet')
cbar = plt.colorbar()
cbar.set_label(r'Probability Density $|\psi(j, t)|^2$')
plt.gca().invert_yaxis()

plt.xlabel('Time t')
plt.ylabel('Position j')
plt.title('Probability Density Heatmap')

plt.savefig('task2_plot.png')
plt.show()


# Output results
print("--N 101 --F 0.4 --k0 0.618 --alpha 0.314 --N0 33 --t_max 100 --t 42 --j_list 10 20 30 40 50--")
print("==========Task 2: Output==========")

print("\n(1)_10_eigenvalues:",eigenvalues[:10])
print("\n(2)_Probability:",prob_values)
print("\n(3)_Plot: figure saved as task2_plot.png")