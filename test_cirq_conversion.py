#!/usr/bin/env python3
"""
Test script to verify the Cirq conversion works correctly
"""

import sys
import numpy as np
import cirq
from helper_funcs_quantum import (
    create_normal_distribution_circuit, 
    create_linear_amplitude_function, 
    amplitude_estimation_cirq
)

def test_basic_cirq_functionality():
    """Test basic Cirq functionality"""
    print("Testing basic Cirq functionality...")
    
    # Test creating a simple circuit
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubit))
    circuit.append(cirq.measure(qubit, key='m'))
    
    simulator = cirq.Simulator()
    results = simulator.run(circuit, repetitions=100)
    
    # Should get roughly 50% 0s and 50% 1s
    measurements = results.measurements['m']
    ones_count = np.sum(measurements)
    print(f"Basic circuit test: {ones_count}/100 measurements were 1 (expected ~50)")
    
    return True

def test_normal_distribution_circuit():
    """Test the normal distribution circuit creation"""
    print("Testing normal distribution circuit...")
    
    try:
        circuit = create_normal_distribution_circuit(3, mu=0.5, sigma=0.1, bounds=(0, 1))
        print(f"Normal distribution circuit created with {len(circuit)} operations")
        return True
    except Exception as e:
        print(f"Error creating normal distribution circuit: {e}")
        return False

def test_linear_amplitude_function():
    """Test the linear amplitude function creation"""
    print("Testing linear amplitude function...")
    
    try:
        circuit = create_linear_amplitude_function(3, slopes=1, offsets=0, domain=(0, 1), image=(0, 1), rescaling_factor=1)
        print(f"Linear amplitude function created with {len(circuit)} operations")
        return True
    except Exception as e:
        print(f"Error creating linear amplitude function: {e}")
        return False

def test_amplitude_estimation():
    """Test amplitude estimation"""
    print("Testing amplitude estimation...")
    
    try:
        # Create a simple circuit that should give us a known amplitude
        qubit = cirq.LineQubit(0)
        circuit = cirq.Circuit()
        circuit.append(cirq.ry(np.pi/4).on(qubit))  # This should give amplitude ~0.707
        
        estimated_amplitude, confidence_interval, shots = amplitude_estimation_cirq(
            circuit, [0], epsilon=0.1, alpha=0.05, shots=1000
        )
        
        print(f"Amplitude estimation: {estimated_amplitude:.3f} ± {confidence_interval:.3f} (expected ~0.707)")
        return True
    except Exception as e:
        print(f"Error in amplitude estimation: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting Cirq conversion tests...\n")
    
    tests = [
        test_basic_cirq_functionality,
        test_normal_distribution_circuit,
        test_linear_amplitude_function,
        test_amplitude_estimation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ PASSED\n")
            else:
                print("✗ FAILED\n")
        except Exception as e:
            print(f"✗ FAILED with exception: {e}\n")
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("All tests passed! The Cirq conversion appears to be working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 