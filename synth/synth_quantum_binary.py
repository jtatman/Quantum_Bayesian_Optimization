import sys
sys.path.append('../')

from bayesian_optimization_quantum import QBO
import pickle
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import cirq
from helper_funcs_quantum import amplitude_estimation_cirq

quantum_noise = False
binary = True # always set this to True
linear_bandit = False

max_iter = int(1e4)

ls = 0.1

obs_noise = 0.4**2
log_file_name = "saved_synth_funcs/synth_func_ls_" + str(ls) + "_noise_var_" + str(obs_noise) + ".pkl"
all_func_info = pickle.load(open(log_file_name, "rb"))
domain = all_func_info["domain"]
f = all_func_info["f"]

log_file_name = "saved_synth_funcs/random_features_ls_" + str(ls) + "_noise_var_" + str(obs_noise) + ".pkl"
random_features = pickle.load(open(log_file_name, "rb"))

def synth_func(param, eps):
    x = param[0]
    ind = np.argmin(np.abs(domain - x))
    
    num_uncertainty_qubits = 1
    
    p = f[ind, 0]
    low = 0
    high = 1
    
    # Calculate the rotation angle
    theta = 2 * np.arcsin(np.sqrt(p))
    
    # Create a quantum circuit using Cirq
    qubit = cirq.LineQubit(0)
    qc = cirq.Circuit()
    
    # Apply the RY rotation
    qc.append(cirq.ry(theta).on(qubit))
    uncertainty_model = qc

    monte_carlo = uncertainty_model

    epsilon = eps
    epsilon = np.clip(epsilon, 1e-6, 0.5)

    alpha = 0.05
    max_shots = int(32 * np.log(2/alpha*np.log2(np.pi/(4*epsilon))))
    
    # Define objective qubits (first qubit in this case)
    objective_qubits = [0]

    # Perform amplitude estimation using Cirq
    estimated_amplitude, confidence_interval, num_oracle_queries = amplitude_estimation_cirq(
        monte_carlo, 
        objective_qubits, 
        epsilon, 
        alpha, 
        shots=max_shots
    )

    est = estimated_amplitude

    if num_oracle_queries == 0:
        # use the number of oracle calls given by the paper if num_oracle_queries == 0
        num_oracle_queries = int(np.ceil((0.8 / epsilon) * np.log((2 / alpha) * np.log2(np.pi / (4 * epsilon)))))

    return est, p, num_oracle_queries

ts = np.arange(1, max_iter)
beta_t = 1 + np.sqrt(np.log(ts) ** 2)

run_list = np.arange(10)
for itr in run_list:
    np.random.seed(itr)

    log_file_name = "results_quantum/res_iter_" + str(itr) + "_binary.pkl"

    if linear_bandit:
        log_file_name = log_file_name[:-4] + "_linear_bandit.pkl"
    if quantum_noise:
        log_file_name = log_file_name[:-4] + "_quantum_noise.pkl"

    quantum_BO = QBO(f=synth_func, pbounds={'x1':(0, 1)}, log_file=log_file_name, beta_t=beta_t, \
              random_features=random_features, linear_bandit=linear_bandit, domain=domain)
    quantum_BO.maximize(n_iter=max_iter, init_points=1)
