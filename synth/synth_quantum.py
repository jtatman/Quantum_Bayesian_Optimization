import sys
sys.path.append('../')

from bayesian_optimization_quantum import QBO
import pickle
import numpy as np
import cirq
from helper_funcs_quantum import create_normal_distribution_circuit, create_linear_amplitude_function, amplitude_estimation_cirq

quantum_noise = False # whether to consider quantum noise
linear_bandit = False # whether to run quantum linear bandit algorithm; set it to False by default

max_iter = int(1e4)

ls = 0.1

obs_noise = 0.3**2
# obs_noise = 0.4**2

log_file_name = "saved_synth_funcs/synth_func_ls_" + str(ls) + "_noise_var_" + str(obs_noise) + ".pkl"
all_func_info = pickle.load(open(log_file_name, "rb"))
domain = all_func_info["domain"]
f = all_func_info["f"]

log_file_name = "saved_synth_funcs/random_features_ls_" + str(ls) + "_noise_var_" + str(obs_noise) + ".pkl"
random_features = pickle.load(open(log_file_name, "rb"))

def synth_func(param, eps):
    x = param[0]
    ind = np.argmin(np.abs(domain - x))

    num_uncertainty_qubits = 6

    mean = f[ind, 0]
    variance = obs_noise
    stddev = np.sqrt(variance)

    low = mean - 3 * stddev
    high = mean + 3 * stddev

    # Create uncertainty model using Cirq
    uncertainty_model = create_normal_distribution_circuit(num_uncertainty_qubits, mean, stddev**2, (low, high))
    
    c_approx = 1
    slopes = 1
    offsets = 0
    f_min = low
    f_max = high

    # Create linear amplitude function using Cirq
    linear_payoff = create_linear_amplitude_function(
        num_uncertainty_qubits,
        slopes,
        offsets,
        (low, high),
        (f_min, f_max),
        c_approx,
    )

    # Combine uncertainty model and linear payoff
    # In Cirq, we can combine circuits by adding them
    monte_carlo = uncertainty_model + linear_payoff

    # Set target precision and confidence level
    epsilon = eps / (3 * stddev)
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
    
    # Post-processing: map the amplitude back to the original domain
    est = estimated_amplitude * (high - low) + low

    if num_oracle_queries == 0:
        # use the number of oracle calls given by the paper if num_oracle_queries == 0
        num_oracle_queries = int(np.ceil((0.8 / epsilon) * np.log((2 / alpha) * np.log2(np.pi / (4 * epsilon)))))

    return est, mean, num_oracle_queries

ts = np.arange(1, max_iter)
beta_t = 1 + np.sqrt(np.log(ts) ** 2)

run_list = np.arange(10)
for itr in run_list:
    np.random.seed(itr)
    
    log_file_name = "results_quantum/res_noise_var_" + str(obs_noise) + "_iter_" + str(itr) + ".pkl"

    if linear_bandit:
        log_file_name = log_file_name[:-4] + "_linear_bandit.pkl"
    if quantum_noise:
        log_file_name = log_file_name[:-4] + "_quantum_noise.pkl"

    quantum_BO = QBO(f=synth_func, pbounds={'x1':(0, 1)}, log_file=log_file_name, beta_t=beta_t, \
              random_features=random_features, linear_bandit=linear_bandit, domain=domain)
    quantum_BO.maximize(n_iter=max_iter, init_points=1)
