""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle
import math
import sys
import random


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        eta= 0
        loc_particles= np.zeros((self.particles.shape))
        loc_weights= np.ones(self.num_particles)/self.num_particles
        for i in range(self.num_particles):
            loc_particles[i,:] = env.forward(self.particles[i,:],env.sample_noisy_action(u)).reshape(self.particles[i,:].shape)# use noisy input
            z_hat= env.sample_noisy_observation(loc_particles[i,:],marker_id)

            loc_weights[i]*= env.likelihood(z_hat-z, self.beta)+1e-300

        loc_weights= loc_weights/np.sum(loc_weights)
        self.particles,self.weights= self.resample(loc_particles, loc_weights)
        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """

        r= np.random.rand()*self.num_particles**-1
        c= weights[0]
        weights[-1]=1
        new_particles=[]

        i=0
        for m in range(0,self.num_particles):
            U = r+ (m)/(self.num_particles)

            while(U>c):
                i+=1
                c+= weights[i]
            new_particles.append(particles[i])
        new_particles= np.asarray(new_particles)    
        assert new_particles.shape==particles.shape

        # current=0
        # cum_prob=[]
        # new_particles=[]
        # for i in weights:
        #     current+=i
        #     cum_prob.append(current)
        # cum_prob[-1]=1# to make sure the last one is 1 not 0.999999 etc.
        #
        # spokes= np.linspace(0,1,self.num_particles)
        #
        # j=0
        # for i in spokes:
        #     while(i>cum_prob[j]):
        #         #print(i,", ",cum_prob[j] )
        #         j+=1
        #     new_particles.append(particles[j])
        # new_particles= np.asarray(new_particles)
        return new_particles, weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov
