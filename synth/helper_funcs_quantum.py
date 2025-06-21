import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import pickle
import cirq
import cirq.contrib.qcircuit as qcircuit

def acq_max(ac, M, random_features, bounds, nu_t, Sigma_t_inv, beta, domain, linear_bandit):
    para_dict={"M":M, "random_features":random_features, "nu_t":nu_t, "Sigma_t_inv":Sigma_t_inv, \
              "beta":beta, "linear_bandit":linear_bandit}
    ys = []
    for i, x in enumerate(domain):
        ys.append(-ac(x.reshape(1, -1), para_dict))

    ys = np.squeeze(np.array(ys))
    argmin_ind = np.argmin(ys)
    x_max = domain[argmin_ind, :]

    return x_max

class UtilityFunction(object):
    def __init__(self):
        self.kind = "ucb"
    def utility(self, x, para_dict):
        M, random_features, nu_t, Sigma_t_inv, beta, linear_bandit = para_dict["M"], \
            para_dict["random_features"],\
            para_dict["nu_t"], para_dict["Sigma_t_inv"], para_dict["beta"], \
            para_dict["linear_bandit"]

        if self.kind == 'ucb':
            return self._ucb(x, random_features, nu_t, Sigma_t_inv, beta, linear_bandit)

    @staticmethod
    def _ucb(x, random_features, nu_t, Sigma_t_inv, beta, linear_bandit):
        d = x.shape[1]

        s = random_features["s"]
        b = random_features["b"]
        obs_noise = random_features["obs_noise"]
        v_kernel = random_features["v_kernel"]
        M = b.shape[0]

        if not linear_bandit:
            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
            features = features.reshape(-1, 1)

            features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
            features = np.sqrt(v_kernel) * features

        else:
            features = x.transpose()

        mean = np.squeeze(np.dot(features.T, nu_t))
        
        lam = 1
        var = lam * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))

        std = np.sqrt(var)

        return np.squeeze(mean + beta * std)

def create_normal_distribution_circuit(num_qubits, mu, sigma, bounds):
    """
    Create a Cirq circuit that encodes a normal distribution
    """
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    
    # Create a simple normal distribution approximation using rotations
    # This is a simplified version - in practice you might want a more sophisticated approach
    for i, qubit in enumerate(qubits):
        # Simple rotation based on the mean and variance
        angle = 2 * np.pi * (mu + i * sigma) / (2**num_qubits)
        circuit.append(cirq.ry(angle).on(qubit))
    
    return circuit

def create_linear_amplitude_function(num_qubits, slopes, offsets, domain, image, rescaling_factor):
    """
    Create a Cirq circuit that implements a linear amplitude function
    """
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    
    # Simplified linear amplitude function using rotations
    for i, qubit in enumerate(qubits):
        # Calculate rotation based on linear function parameters
        angle = slopes * (i / (2**num_qubits - 1)) + offsets
        circuit.append(cirq.ry(angle).on(qubit))
    
    return circuit

def amplitude_estimation_cirq(circuit, objective_qubits, epsilon, alpha, shots=1000):
    """
    Perform amplitude estimation using Cirq
    """
    # Create simulator
    simulator = cirq.Simulator()
    
    # Add measurement operations to the circuit
    measured_circuit = circuit.copy()
    for qubit_index in objective_qubits:
        qubit = cirq.LineQubit(qubit_index)
        measured_circuit.append(cirq.measure(qubit, key=f'm_{qubit_index}'))
    
    # Run the circuit multiple times to estimate amplitude
    results = simulator.run(measured_circuit, repetitions=shots)
    
    # Calculate the probability of measuring the objective qubits in state |1⟩
    success_count = 0
    for i in range(shots):
        # Check if all objective qubits are measured as 1
        all_ones = True
        for qubit_index in objective_qubits:
            if results.measurements[f'm_{qubit_index}'][i] != 1:
                all_ones = False
                break
        if all_ones:
            success_count += 1
    
    # Estimate the amplitude
    estimated_amplitude = np.sqrt(success_count / shots)
    
    # Calculate confidence interval
    confidence_interval = 1.96 * np.sqrt(estimated_amplitude * (1 - estimated_amplitude) / shots)
    
    return estimated_amplitude, confidence_interval, shots

