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
        self.set_xlim(-np.pi, np.pi)
        self.ymin = self.ymax = None

    def set_size(self, N):
        self.N = np.array(N)
        i, = (self.N == 0).nonzero()
        if len(i) > 1:
            raise ValueError("Too many open directions.")
        self.open = i[0]

    def set_xlim(self, xmin: float, xmax: float):
        self.xmin, self.xmax = xmin, xmax
        self.mesh = np.linspace(self.xmin, self.xmax, self.sim_density)

    def set_ylim(self, ymin: float, ymax: float):
        self.ymin, self.ymax = ymin, ymax

    def _compute(self, f, i):
        kvec = [0.0] * len(self.N)
        kvec[self.open] = self.mesh[i]
        return scipy.linalg.eigvalsh(self.model(self.xlim[f]).open_hamiltonian(self.N, kvec))
        # self.bands[f, i, :] = scipy.linalg.eigvalsh(self.model(self.xlim[f]).open_hamiltonian(self.N, kvec))

    def generate(self, max_cpu: int=0):
        if max_cpu == 0:
            max_cpu = mp.cpu_count() // 2
        # try:
        # self.bands = np.zeros((len(self.xlim), self.sim_density, self.model(self.xlim[0]).open_hamiltonian(self.N, (0.0,) * len(self.N)).shape[0]))
        F, I = np.meshgrid(np.arange(len(self.xlim)), np.arange(self.sim_density))
        with mp.Pool(max_cpu) as pool:
            result = pool.starmap(self._compute, zip(F.flatten(), I.flatten()))
        self.bands = np.array(result).reshape(self.sim_density, len(self.xlim), len(result[0])).transpose((1, 0, 2))
        # except AttributeError:
        #     raise AttributeError("Size of the open system is not set.")
        

    def plot(self, filename, title=""):
        fig = plt.figure()
        ax = fig.gca()

        if self.ymin is not None and self.ymax is not None:
            ax.set_ylim(self.ymin, self.ymax)
        ax.set_xlim(self.xmin, self.xmax)
        dim_labels = lambda i: ["x", "y", "z", "w"][i] if i <= 3 else str(i)
        ax.set_xlabel("$k_{" + dim_labels(self.open) + "}$")
        ax.set_ylabel("$E$")
        if title:
            ax.set_title(title)

        lines = [0] * self.bands.shape[-1]
        for i in range(self.bands.shape[-1]):
            lines[i], = ax.plot(self.mesh, self.bands[0, :, i], "k-")

        def init_plot():
            for i in range(self.bands.shape[-1]):
                lines[i].set_data(self.mesh, self.bands[0, :, i])
            return lines

        def plot_frame(frame):
            for i in range(self.bands.shape[-1]):
                lines[i].set_data(self.mesh, self.bands[frame, :, i])
            ax.set_title(title % self.xlim[frame])
            return lines

        a = anim.FuncAnimation(fig, plot_frame, init_func=init_plot, frames=len(self.xlim), interval=200, blit=False)
        writer = anim.PillowWriter(fps=self.fps, bitrate=1800)
        a.save(filename, writer=writer)
