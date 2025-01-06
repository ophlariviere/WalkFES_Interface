import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from skopt import gp_minimize
from skopt.space import Real

import biorbd
from pysciencemode import Modes


class BayesianOptimizer():
    def __init__(self, visualization_widget):

        self.visualization_widget = visualization_widget

        # Define the model
        self.model = visualization_widget.model

        # Define the variable bounds
        self.bounds = [
            Real(20, 50, name='R_frequency'),  # Hz
            Real(8, 20, name='R_intensity'),  # mA
            Real(200, 500, name='R_width'),  # micros
            Real(20, 50, name='L_frequency'),  # Hz
            Real(8, 20, name='L_intensity'),  # mA
            Real(200, 500, name='L_width'),  # micros
        ]

        # Define the objective weightings
        # TODO: Charbie -> how do we chose which objectives to minimize ?
        self.weight_comddot = 1
        self.weight_angular_momentum = 1
        self.weight_enegy = 1


    def make_an_iteration(self, params):

        # Current values of the optimized FES parameters
        R_frequency = params[0]
        R_intensity = params[1]
        R_width = params[2]
        L_frequency = params[3]
        L_intensity = params[4]
        L_width = params[5]

        # Send FES with these parameter values
        # Setup Right leg
        self.visualization_widget.set_chanel_inputs(channel=0,
                                                    channel_layout=None,
                                                    name_input="Right",
                                                    amplitude_input=R_intensity,  # TODO: @ophelielariviere, is it the right parameter ?
                                                    pulse_width_input=R_width,
                                                    frequency_input=R_frequency,
                                                    mode_input=Modes.SINGLE)
        # Setup Left leg
        self.visualization_widget.set_chanel_inputs(channel=1,
                                                    channel_layout=None,
                                                    name_input="Left",
                                                    amplitude_input=L_intensity,  # TODO: @ophelielariviere, is it the right parameter ?
                                                    pulse_width_input=L_width,
                                                    frequency_input=L_frequency,
                                                    mode_input=Modes.SINGLE)
        # Stimulate
        self.visualization_widget.start_stimulation(self, [0, 1])

        #
        cycles = {'StanceDuration_L': np.zeros((1, 1)),
            'StanceDuration_R': np.zeros((1, 1)),
            'Cycleduration': np.zeros((1, 1)),
            'StepWidth': np.zeros((1, 1)),
            'StepLength_L': np.zeros((1, 1)),
            'StepLength_R': np.zeros((1, 1)),
            'PropulsionDuration_L': np.zeros((1, 1)),
            'PropulsionDuration_R': np.zeros((1, 1)),
            'Cadence': np.zeros((1, 1)),
                  }
         # TODO: Charbie -> Rendue à travailler ici pour édtecter quand on est stable :)
        old_cycledata = None
        new_cycledata = "Fake_data"
        while num_stable_cycle < 10:
            while old_cycledata == new_cycledata:
                new_cycledata = self.visualization_widget.buffer.get()
                std_new_cycledata['gait_parameter']
                new_cycledata['Tau']
                new_cycledata['VitAng']
                new_cycledata['Angle']
                new_cycledata['AccAng']
                time.sleep(0.1)
            cycles.append(new_cycledata)
            old_cycledata = new_cycledata



    @staticmethod
    def compute_com_acceleration(model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray):

        nb_frames = q.shape[1]

        comddot = np.zeros((nb_frames,))
        for i_frame in range(nb_frames):
            comddot[i_frame] = np.linalg.norm(
                model.CoMddot(q[:, i_frame], qdot[:, i_frame], qddot[:, i_frame]).to_array())

        return comddot

    @staticmethod
    def compute_angular_momentum(model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray):

        nb_frames = q.shape[1]

        angular_momentum = np.zeros((nb_frames,))
        for i_frame in range(nb_frames):
            angular_momentum[i_frame] = np.linalg.norm(
                model.angularMomentum(q[:, i_frame], qdot[:, i_frame]).to_array())

        return angular_momentum

    @staticmethod
    def compute_energy(qdot, tau, R_intensity, L_intensity, time_vector):
        """
        Since the time is the same, min energy and power gives the same thing (same min).
        """

        voltage = 30  # TODO: @ophelielariviere, what is the voltage ?
        power_stim = np.abs(R_intensity * voltage) + np.abs(L_intensity * voltage)
        power_total = np.sum(np.abs(tau * qdot), axis=0)
        power_human = power_total - power_stim
        energy_human = np.trapezoid(power_human, x=time_vector)

        return energy_human

    def objective(self, model, params):

        # Receive data... TODO
        nb_frames = 100  # Let's say 100 frames
        q = np.zeros((16 * 3, nb_frames))
        qdot = np.zeros((16 * 3, nb_frames))
        qddot = np.zeros((16 * 3, nb_frames))
        tau = np.zeros((16 * 3, nb_frames))
        time_vector = np.linspace(0, 1, nb_frames)  # Let's say a step of 1s

        comddot = self.compute_com_acceleration(model, q, qdot, qddot)
        angular_momentum = self.compute_angular_momentum(model, q, qdot, qddot)
        energy_human = self.compute_energy(qdot, tau, R_intensity, L_intensity, time_vector)

        return self.weight_comddot * comddot + self.weight_angular_momentum * angular_momentum + self.weight_enegy * energy_human


    def perform_bayesian_optim(self):
        """Perform Bayesian optimization using Gaussian Processes."""

        # Activate the stimulator # TODO: @ophelielariviere: with default values for the parameters ?
        self.visualization_widget.activate_stimulateur()

        # gp_minimize will try to find the minimal value of the objective function.
        result = gp_minimize(
            func =lambda params: self.make_an_iteration(params),
            dimensions=self.bounds,
            n_calls=100,         # number of evaluations of f
            acq_func="LCB",      # "LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"
            kappa=5,  # *
            random_state=0,  #*
            n_jobs=1,
        )  #x0, y0, kappa[exploitation, exploration], xi [minimal improvement default 0.01]
        # TODO: stop when the same point has been hit t time (t=5 in general)
        return result

    def plot_bayesian_optim_results():
        print("Best found minimum:")
        print("X = %.4f, Y = %.4f" % (result.x[0], result.x[1]))
        print("f(x,y) = %.4f" % result.fun)

        # Optionally, plot convergence
        fig = plt.figure(figsize=(12, 5))
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection="3d")

        # Convergence plot
        ax0.plot(result.func_vals, marker='o')
        ax0.set_title('Convergence Plot')
        ax0.set_xlabel('Number of calls')
        ax0.set_ylabel('Objective function value')

        # Plot the function sampling
        x_iters_array = np.array(result.x_iters)
        func_vals_array = np.array(result.func_vals)
        colors_min = np.min(func_vals_array)
        colors_max = np.max(func_vals_array)
        normalized_cmap = (func_vals_array - colors_min) / (colors_max - colors_min)
        colors = cm["viridis"](normalized_cmap)
        p = ax1.scatter(x_iters_array[:, 0], x_iters_array[:, 1], result.func_vals, c=colors, marker='.')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Function Sampling')
        cbar = fig.colorbar(p)
        cbar.set_label('Objective function value')
        plt.show()
