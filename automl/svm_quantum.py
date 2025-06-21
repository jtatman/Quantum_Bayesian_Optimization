import sys
sys.path.append('../')

from bayesian_optimization_quantum import QBO
import pickle
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from sklearn import svm
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import cirq
from helper_funcs_quantum import create_normal_distribution_circuit, create_linear_amplitude_function, amplitude_estimation_cirq

quantum_noise = False
linear_bandit = False

max_iter = int(5000)

ls = np.array([0.2, 0.2]) # this seems to be the best
v_kernel = 0.5
M_target = 200
log_file_name = "saved_synth_funcs/random_features_ls_" + str(ls) + "_v_kernel_" + str(v_kernel) + \
        "_M_" + str(M_target) + ".pkl"
random_features = pickle.load(open(log_file_name, "rb"))
domain = random_features["domain"]

# obs_noise = 0.01**2
obs_noise = 0.05**2

diabetes_data = pd.read_csv("clinical_data/diabetes.csv")
label = np.array(diabetes_data["Outcome"])
features = np.array(diabetes_data.iloc[:, :-1])
X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size=0.3, stratify=label, random_state=0)
n_ft = X_train.shape[1]
n_classes = 2

def svm_reward_function(param, eps):
    parameter_range = [[1e-4, 1.0], [1e-4, 1.0]]
    C_ = param[0]
    C = C_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0]
    gam_ = param[1]
    gam = gam_ * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]

    clf = svm.SVC(kernel="rbf", C=C, gamma=gam, probability=True)
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    acc = np.count_nonzero(pred == Y_test) / len(Y_test)

    num_uncertainty_qubits = 6

    mean = acc
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
    monte_carlo = uncertainty_model + linear_payoff

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

dim = 2
pbounds = {}
for i in range(dim):
    pbounds["x" + str(i+1)] = (0, 1)

ts = np.arange(1, max_iter)
beta_t = 1 + np.sqrt(np.log(ts) ** 2)

run_list = np.arange(5)
for itr in run_list:
    np.random.seed(itr)

    log_file_name = "results_quantum/res_noise_var_" + str(obs_noise) + "_iter_" + str(itr) + ".pkl"

    if linear_bandit:
        log_file_name = log_file_name[:-4] + "_linear_bandit.pkl"
    if quantum_noise:
        log_file_name = log_file_name[:-4] + "_quantum_noise.pkl"

    quantum_BO = QBO(f=svm_reward_function, pbounds=pbounds, log_file=log_file_name, beta_t=beta_t, \
              random_features=random_features, linear_bandit=linear_bandit, domain=domain)
    quantum_BO.maximize(n_iter=max_iter, init_points=1)
