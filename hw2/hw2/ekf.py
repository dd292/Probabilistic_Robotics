""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle
import sys
import copy

class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = copy.deepcopy(self._init_mean)
        self.sigma = copy.deepcopy(self._init_cov)

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE
        # PREDICTION STEP

        G = env.G(self.mu, u)
        V = env.V(self.mu, u)
        M = env.noise_from_motion(u, self.alphas)

        predicted_mean = env.forward(self.mu, u)
        predicted_sigma = G @ self.sigma @ G.T + V @ M @ V.T

        #CORRECTION STEP

        predicted_measurement_mean = env.observe(predicted_mean, marker_id)
        H = env.H(predicted_mean, marker_id).reshape((1,-1))

        pred_measurement_cov = H @ predicted_sigma @ H.T + np.diag(self.beta)

        kalamn_gain= predicted_sigma @ H.T @ np.linalg.inv(pred_measurement_cov)

        self.mu= predicted_mean + kalamn_gain @ (minimized_angle(z-predicted_measurement_mean))
        interm = kalamn_gain @ H

        self.sigma= (np.eye(interm.shape[0])- interm) @ predicted_sigma
        return self.mu, self.sigma
