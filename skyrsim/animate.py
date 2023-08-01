import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import multiprocessing as mp
import typing

from .phase import ModelWrapper
# from .edge import EdgeSimulator

class SpectrumAnimator:
    def __init__(self, model: ModelWrapper, xlim: typing.Union[tuple, np.ndarray], sim_density: int=101, fps: int=10): 
        self.model = model

        if isinstance(xlim, tuple):
            if len(xlim) != 3:
                raise ValueError("Expected a tuple of length 3 for `xlim`.")
            self.xlim = np.linspace(xlim[0], xlim[1], xlim[2])
        else:
            self.xlim = xlim

        self.sim_density = sim_density
        self.fps = fps
        self.ymin = self.ymax = None
        self.xmin = -np.pi
        self.xmax = np.pi

    def set_size(self, N):
        self.N = np.array(N)
        i, = (self.N == 0).nonzero()
        if len(i) > 1:
            raise ValueError("Too many open directions.")
        self.open = i[0]

    def set_xlim(self, xmin, xmax):
        self.xmin, self.xmax = xmin, xmax

    def set_ylim(self, ymin, ymax):
        self.ymin, self.ymax = ymin, ymax

    def plot(self, filename):
        fig = plt.figure()
        ax = fig.gca()

        if self.ymin is not None and self.ymax is not None:
            ax.set_ylim(self.ymin, self.ymax)
        ax.set_xlim(self.xmin, self.xmax)
        dim_labels = lambda i: ["x", "y", "z", "w"][i] if i <= 3 else str(i)
        ax.set_xlabel("$k_{" + dim_labels(self.open) + "}$")
        ax.set_ylabel("$E$")

        # try:
        bands = np.zeros((self.sim_density, self.model(self.xlim[0]).open_hamiltonian(self.N, (0.0,) * len(self.N)).shape[0]))
        kvec = [0.0] * len(self.N)
        lines = [0] * bands.shape[1]
        # except AttributeError:
            # raise AttributeError("Size of the open system is not set.")

        for i in range(bands.shape[1]):
            lines[i], = ax.plot([], [], "k-")

        def init_plot():
            for i in range(bands.shape[1]):
                lines[i].set_data([], [])
            return lines

        mesh = np.linspace(self.xmin, self.xmax, self.sim_density)
        def plot_frame(frame):
            for i, k in zip(range(self.sim_density), mesh):
                kvec[self.open] = k
                bands[i, :] = scipy.linalg.eigvalsh(self.model(self.xlim[frame]).open_hamiltonian(self.N, kvec))
            for i in range(bands.shape[1]):
                lines[i].set_data(mesh, bands[:, i])
            return lines

        a = anim.FuncAnimation(fig, plot_frame, init_func=init_plot, frames=len(self.xlim), interval=200, blit=False)
        writer = anim.PillowWriter(fps=self.fps, bitrate=1800)
        a.save(filename, writer=writer)
