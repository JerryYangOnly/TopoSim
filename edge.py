import numpy as np
import scipy
import matplotlib.pyplot as plt

from bulk import Simulator
from model import *

class EdgeSimulator(Simulator):
    def __init__(self, model: Model, points: int):
        super().__init__(model, points)

        self.N = (0) * self.model.dim
        self.open_dim = []
        self.eff_dim = self.model.dim
        self.eff_bands = self.model.bands

    def open(self, N):
        self.N = np.array(N)
        self.open_dim = np.arange(self.model.dim)[self.N > 0]
        
        self.eff_dim = self.model.dim - len(self.open_dim)
        self.eff_bands = self.model.bands * np.prod(self.N[self.open_dim])
        self.band = np.zeros((*([self.mesh_points] * self.eff_dim), self.eff_bands))
        self.states = np.zeros((*([self.mesh_points] * self.eff_dim), self.eff_bands, self.eff_bands), dtype=np.complex64)
         
        self.evaluated = False

    def populate_mesh(self):
        if self.evaluated:
            return

        for i in range(self.mesh_points**self.eff_dim):
            idx = [0] * self.model.dim
            k = i
            for j in range(self.model.dim - 1, -1, -1):
                if j not in self.open_dim:
                    idx[j] = k % self.mesh_points
                    k //= self.mesh_points
            w, v = scipy.linalg.eigh(self.model.open_hamiltonian(self.N, [self.mesh[j] for j in idx]))
            self.band[i, :] = w
            self.states[i, :, :] = v

        self.band = self.band.reshape((*([self.mesh_points] * self.eff_dim), self.eff_bands))
        self.states = self.states.reshape((*([self.mesh_points] * self.eff_dim), self.eff_bands, self.eff_bands))
        self.evaluated = True

    def plot_band(self, filled_bands=None, full=False):
        if not self.evaluated:
            self.populate_mesh()
        if not filled_bands:
            filled_bands = self.eff_bands // 2    # Default to half-filling

        if self.eff_dim == 1:
            fig, ax = plt.subplots()
            if not full:
                ax.plot(self.mesh, self.band[:, filled_bands - 1])
                ax.plot(self.mesh, self.band[:, filled_bands])
            else:
                for i in range(self.eff_bands):
                    ax.plot(self.mesh, self.band[:, i])
            plt.show()

        elif self.eff_dim == 2:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            X, Y = np.meshgrid(self.mesh, self.mesh)

            if not full:
                ax.plot_surface(X, Y, self.band[:, :, filled_bands - 1])
                ax.plot_surface(X, Y, self.band[:, :, filled_bands])
            else:
                for i in range(self.eff_bands):
                    ax.plot_surface(X, Y, self.band[:, :, i])
            del X, Y
            plt.show()

        else:
            print("Band plotting of models in %d-D is not supported." % self.model.dim)


for u in np.arange(-2.5, 2.6, 1):
    sim = EdgeSimulator(QWZModel(u=u), 41)
    sim.open((10, 0))
    sim.plot_band(full=True)
