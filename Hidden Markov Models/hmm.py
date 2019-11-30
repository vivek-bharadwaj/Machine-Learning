from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, obs_sequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_sequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(obs_sequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        # Base Case of Recursion
        alpha[:, 0] = self.pi * self.B[:, self.obs_dict[obs_sequence[0]]]

        # for t in range 2 -> N
        for t in range(1, L):
            alpha[:, t] = self.B[:, self.obs_dict[obs_sequence[t]]] * np.matmul(np.transpose(self.A), alpha[:, t-1])
        ###################################################
        return alpha

    def backward(self, obs_sequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_sequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(obs_sequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        beta[:, L-1] = 1

        for t in reversed(range(0, L-1)):
            beta[:, t] = np.matmul(self.A, np.multiply(self.B[:, self.obs_dict[obs_sequence[t + 1]]], beta[:, t + 1]))
        ###################################################
        return beta

    def sequence_prob(self, obs_sequence):
        """
        Inputs:
        - obs_sequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        return prob

    def posterior_prob(self, obs_sequence):
        """
        Inputs:
        - obs_sequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(obs_sequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        return prob

    # TODO:
    def likelihood_prob(self, obs_sequence):
        """
        Inputs:
        - obs_sequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(obs_sequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        ###################################################
        return prob

    def viterbi(self, obs_sequence):
        """
        Inputs:
        - obs_sequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        ###################################################
        return path
