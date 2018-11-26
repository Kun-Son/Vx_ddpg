
import numpy as np

class ExplorationNoise(object):

    # ================================
    #    WHITE NOISE PROCESS
    # ================================

    def white_noise(mu, sigma, num_steps):
        # Generate random noise with mean 0 and variance 1
        return np.random.normal(mu, sigma, num_steps)

    # ================================
    #    ORNSTEIN-UHLENBECK PROCESS
    # ================================

    def ou_noise(theta, mu, sigma, num_steps, dt=1.):
        noise = np.zeros(num_steps)

        # Generate random noise with mean 0 and variance 1
        white_noise = np.random.normal(0, 1, num_steps)

        # Solve using Euler-Maruyama method
        for i in range(1, num_steps):
            noise[i] = noise[i - 1] + theta * (mu - noise[i - 1]) * \
                                                dt + sigma * np.sqrt(dt) * white_noise[i]

        return noise

    # ================================
    #    EXPONENTIAL NOISE DECAY
    # ================================

    def exp_decay(noise, decay_end):
        num_steps = noise.shape[0]
        # Check if decay ends before end of noise sequence
        assert(decay_end <= num_steps)

        scaling = np.zeros(num_steps)

        scaling[:decay_end] = 2. - np.exp(np.divide(np.linspace(1., decay_end, num=decay_end) * np.log(2.), decay_end))

        return np.multiply(noise, scaling)

    # ================================
    #    TANH NOISE DECAY
    # ================================

    def tanh_decay(noise, decay_start, decay_length):
        num_steps = noise.shape[0]
        # Check if decay ends before end of noise sequence
        assert(decay_start + decay_length <= num_steps)

        scaling = 0.5*(1. - np.tanh(4. / decay_length * np.subtract(np.linspace(1., num_steps, num_steps),
                                                              decay_start + decay_length/2.)))

        return np.multiply(noise, scaling)