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

    def plot_band(self, band_hl=()):
        if not self.evaluated:
            self.populate_mesh()

        if self.eff_dim == 1:
            fig, ax = plt.subplots()
            for i in range(self.eff_bands):
                if i not in band_hl:
                    ax.plot(self.mesh, self.band[:, i], "k-")
                else:
                    ax.plot(self.mesh, self.band[:, i])
            plt.show()

        elif self.eff_dim == 2:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            X, Y = np.meshgrid(self.mesh, self.mesh)

            for i in range(self.eff_bands):
                ax.plot_surface(X, Y, self.band[:, :, i])
            del X, Y
            plt.show()

        else:
            print("Band plotting of models in %d-D is not supported." % self.model.dim)

    def in_gap_states(self, n_states=2, fermi=0.0):
        ids = np.argpartition(np.abs(self.band - fermi), n_states, axis=None)
        ids = np.array(np.unravel_index(ids[:n_states], self.band.shape)).transpose()
        return [self.states.__getitem__(*idx[:-1])[:, idx[-1]] for idx in ids]

    def pdf(self, state, sum_internal=True):
        s = state.reshape(tuple(self.N[self.open_dim]) + (self.model.bands,), order='C')
        s = np.abs(s)**2
        # s /= np.sum(s)      # Normalization

        if sum_internal:
            s = np.sum(s, axis=len(self.open_dim))
        return s


    def pdfs(self, states, sum_internal=True):
        return [self.pdf(state, sum_internal=sum_internal) for state in states]
        
    def position_heat_map_band(self, band):
        if self.eff_dim != 1 or len(self.open_dim) != 1:
            print("Dimension of the model is not supported. Are boundaries opened correctly?")
            return
        if not self.evaluated:
            self.populate_mesh()
        
        fig = plt.figure()
        # ax = fig.gca(projection="3d")
        ax = fig.gca()

        pdfs = np.array(self.pdfs(self.states[:, :, band]))

        ax.imshow(pdfs.transpose(), aspect=2 * np.pi / np.prod(self.N[self.open_dim]), extent=(-np.pi, np.pi, 0, np.prod(self.N[self.open_dim])))
        ax.set_xlabel("Momentum")
        ax.set_ylabel("Site")
        # ax.set_zlim(0, 1)

        plt.show()
        
    def plot_spin_band(self, band):
        if self.eff_dim != 1:
            print("Spin plotting of bands is only supported in 1-D.")
            return
        if self.S is None:
            return
        if not self.evaluated:
            self.populate_mesh()

        states = self.states[..., band].reshape((self.mesh_points, np.prod(self.N[self.open_dim]), self.model.bands))
        spin = np.tensordot(self.S, np.conj(states, -1, -2) @ states, ([1, self.eff_dim], [2, self.eff_dim + 1])).transpose().real

        for i in range(3):
            plt.plot(self.mesh, spin[:, i])
            plt.show()

    def entanglement_spectrum(self, filled_bands=None):
        if self.eff_dim != 1:
            print("Entanglement spectra is only supported in 1-D.")
            return

        filled_bands = filled_bands if filled_bands else self.eff_bands // 2
        proj = self.gs_projector(filled_bands)
        proj = proj[:, :self.eff_bands // 2, :self.eff_bands // 2]
        w, _ = np.linalg.eigh(proj)

        for i in range(self.eff_bands // 2):
            plt.plot(self.mesh, w[:, i], "k-")
        plt.show()

        return w
