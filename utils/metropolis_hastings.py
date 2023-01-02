import random
import math

import numpy as np

def metropolis_hastings(target_distribution, proposal_distribution, initial_state, num_samples):
    """
    Generates samples from a target distribution using the Metropolis-Hastings algorithm.

    Parameters:
    - target_distribution: a function that returns the probability of a state under the target distribution
    - proposal_distribution: a function that returns a proposed state given the current state
    - initial_state: the starting state for the Markov chain
    - num_samples: the number of samples to generate

    Returns:
    - a list of samples from the target distribution
    """
    samples = []
    state = initial_state
    
    for _ in range(num_samples):
        proposed_state = proposal_distribution(state)
        acceptance_probability = min(1, target_distribution(proposed_state) / target_distribution(state))
        if random.random() < acceptance_probability:
            state = proposed_state
        samples.append(state)
    return samples

def binomial_distribution(n, p):
    
    def P(k):
        return math.comb(n, k) * p ** k * (1 - p) ** (n - k)
    
    return P

def propose_state(current_state, P):
    
    # Look up the row of the matrix corresponding to the current thickness
    row = P[current_state]

    # Generate a random number between 0 and 1
    r = random.random()
    
    num_states = P.shape[0]
    
    # Determine the next thickness based on the probabilities in the row
    for i in range(num_states):
        if r < sum(row[:i+1]):
            return i
    
    # If the random number is greater than the sum of the probabilities in the row, return the last state
    return num_states - 1

def probability_matrix(num_states, c):
    
    P = np.zeros((num_states, num_states), np.float32)
    
    d = (1 - c) / 2
    chunk = np.array([d, c, d])
    
    for k in range(1, num_states - 1):
        P[k, k - 1: k + 2] = chunk
    
    P[0, :2] = [c, d]
    P[-1, -2:] = [d, c]
    return P