import numpy as np
import scipy
import matplotlib.pyplot as plt
from model import *
from tqdm import tqdm

# A class for simulating results
class Simulator:
    def __init__(self, model: Model, mesh_points: int):
        self.model = model
        self.set_mesh(mesh_points)
        self.set_spin_op(None)

    def set_mesh(self, mesh_points):
        self.mesh_points = mesh_points
        self.mesh = np.linspace(-np.pi, np.pi, mesh_points) #, endpoint=False)
        # self.meshX, self.meshY = np.meshgrid(np.linspace(-np.pi, np.pi, mesh_points, endpoint=False))
        self.evaluated = False
        self.band = np.zeros((*([mesh_points] * self.model.dim), self.model.bands))
        self.states = np.zeros((*([mesh_points] * self.model.dim), self.model.bands, self.model.bands), dtype=np.complex64)

    def populate_mesh(self):
        if self.evaluated:
            return False
        # print("Starting evaluation")
        self.band = self.band.reshape(self.mesh_points**self.model.dim, self.model.bands)
        self.states = self.states.reshape(self.mesh_points**self.model.dim, self.model.bands, self.model.bands)
        for i in range(self.mesh_points**self.model.dim):
            idx = [0] * self.model.dim
            k = i
            for j in range(self.model.dim - 1, -1, -1):
                idx[j] = k % self.mesh_points
                k //= self.mesh_points
            w, v = scipy.linalg.eigh(self.model.hamiltonian([self.mesh[j] for j in idx]))
            self.band[i, :] = w
            self.states[i, :, :] = v
        self.band = self.band.reshape((*([self.mesh_points] * self.model.dim), self.model.bands))
        self.states = self.states.reshape((*([self.mesh_points] * self.model.dim), self.model.bands, self.model.bands))
        self.evaluated = True
        return True

    def gs_projector(self, filled_bands=None):
        if not self.evaluated:
            self.populate_mesh()
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        return self.states[..., :, :filled_bands] @ np.swapaxes(np.conj(self.states[..., :, :filled_bands]), -1, -2)


    def direct_band_gap(self, filled_bands=None):
        if not self.evaluated:
            self.populate_mesh()
        if not filled_bands:
            filled_bands = self.model.bands // 2    # Default to half-filling
        return np.min(self.band[:, :, filled_bands] - self.band[:, :, filled_bands - 1])

    def plot_band(self, filled_bands=None, full=False):
        if not self.evaluated:
            self.populate_mesh()
        if not filled_bands:
            filled_bands = self.model.bands // 2    # Default to half-filling

        if self.model.dim == 1:
            fig, ax = plt.subplots()
            if not full:
                ax.plot(self.mesh, self.band[:, filled_bands - 1])
                ax.plot(self.mesh, self.band[:, filled_bands])
            else:
                for i in range(self.model.bands):
                    ax.plot(self.mesh, self.band[:, i])
            plt.show()

        elif self.model.dim == 2:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            X, Y = np.meshgrid(self.mesh, self.mesh)

            if not full:
                ax.plot_surface(X, Y, self.band[:, :, filled_bands - 1])
                ax.plot_surface(X, Y, self.band[:, :, filled_bands])
            else:
                for i in range(self.model.bands):
                    ax.plot_surface(X, Y, self.band[:, :, i])
            del X, Y
            plt.show()

        else:
            print("Band plotting of models in %d-D is not supported." % self.model.dim)


    def compute_chern(self, filled_bands=None):
        if self.model.dim != 2:
            print("Computation of Chern numbers is only supported in 2-D.")
        else:
            if not self.evaluated:
                self.populate_mesh()
            return self._chern_hatsugai(filled_bands if filled_bands else self.model.bands // 2)

    def _chern_hatsugai(self, filled_bands):
        Q = 0.0
        F = np.conj(self.states[:-1, :-1, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[1:, :-1, :, :filled_bands]
        F = F @ (np.conj(self.states[1:, :-1, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[1:, 1:, :, :filled_bands])
        F = F @ (np.conj(self.states[1:, 1:, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[:-1, 1:, :, :filled_bands])
        F = F @ (np.conj(self.states[:-1, 1:, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[:-1, :-1, :, :filled_bands])
        F = -np.angle(np.linalg.det(F))
        Q = np.sum(F) / 2.0 / np.pi
        return Q

    def compute_z2(self, filled_bands=None, SOC=True):
        if self.model.dim != 2:
            print("Computation of Z2 invariant is only supported in 2-D.")
            return

        if not filled_bands:
            filled_bands = self.model.bands // 2

        if not SOC:
            if filled_bands and filled_bands % 2 == 1:
                print("Error: filled_bands cannot be odd in Z2 computation!")
                return -1
            model_s1 = Model()
            model_s2 = Model()
            model_s1.hamiltonian = lambda k: self.model.hamiltonian(k)[:self.model.bands // 2, :self.model.bands // 2]
            model_s2.hamiltonian = lambda k: self.model.hamiltonian(k)[self.model.bands // 2:, self.model.bands // 2:]
            model_s1.dim = model_s2.dim = self.model.dim
            model_s1.bands = model_s2.bands = self.model.bands // 2
            sim_s1 = Simulator(model_s1, self.mesh_points)
            sim_s2 = Simulator(model_s2, self.mesh_points)
            v = ((sim_s1.compute_chern(filled_bands // 2) - sim_s2.compute_chern(filled_bands // 2)) / 2) % 2
            del model_s1, model_s2, sim_s1, sim_s2
            return v
        else:
            # return self._z2_fu_kane(filled_bands)
            return self._z2_wcc(filled_bands)

    def _z2_fu_kane(self, filled_bands):
        # Method of Fu and Kane
        if self.mesh_points % 2 == 0:
            self.set_mesh(self.mesh_points + 1)
        if not self.evaluated:
            self.populate_mesh()

        # Compute Berry flux for half of the Brillouin zone
        Q = 0.0
        n = self.mesh_points // 2
        F = np.conj(self.states[n:-1, :-1, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[n + 1:, :-1, :, :filled_bands]
        F = F @ (np.conj(self.states[n + 1:, :-1, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[n + 1:, 1:, :, :filled_bands])
        F = F @ (np.conj(self.states[n + 1:, 1:, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[n:-1, 1:, :, :filled_bands])
        F = F @ (np.conj(self.states[n:-1, 1:, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[n:-1, :-1, :, :filled_bands])
        F = -np.angle(np.linalg.det(F))
        Q = np.sum(F)

        # Compute integral of Berry connection about edge of effective Brillouin zone
        def eff_bz_loop(x):
            v = 6 * np.pi
            if x <= 1 / 3:
                return (np.pi, -np.pi + v * x)
            elif x <= 1 / 2:
                return (np.pi - v * (x - 1 / 3), np.pi)
            elif x <= 5 / 6:
                return (0, np.pi - v * (x - 1 / 2))
            else:
                return (v * (x - 5 / 6), -np.pi)
        W = self.wilson_loop(eff_bz_loop, filled_bands=filled_bands)
        print(np.angle(W))
        W = np.angle(scipy.linalg.det(W))
        
        return ((W - Q) / 2 / np.pi) % 2

    def _z2_wcc(self, filled_bands):
        phases_0 = self.wilson_loop(lambda x: (2 * np.pi * (x - 1/2), 0.001), filled_bands=filled_bands, phases=True)
        phases_pi = self.wilson_loop(lambda x: (2 * np.pi * (x - 1/2), np.pi - 0.001), filled_bands=filled_bands, phases=True)
        theta = np.random.rand() * 2 * np.pi - np.pi

        return (sum(phases_0 < theta) - sum(phases_pi < theta)) % 2


    def wilson_loop(self, loop, points=100, filled_bands=None, phases=False):
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        W = np.eye(self.model.bands, dtype=np.complex64)
        origin = None
        for p in np.linspace(0, 1, points, endpoint=False):
            w, v = scipy.linalg.eigh(self.model.hamiltonian(loop(p)))
            if p == 0.0:
                origin = v[:, :filled_bands]
            else:
                P = sum([np.outer(v[:, i], np.conj(v[:, i])) for i in range(filled_bands)])
                W = P @ W
        W = np.conj(origin).T @ W @ origin
        return W if not phases else np.sort(np.angle(scipy.linalg.eig(W)[0]))# np.sort(np.angle(scipy.linalg.eig(self.wilson_loop(loop, points, filled_bands))[0]))

    
    def set_spin_op(self, S):
        self.S = S
        self.spin_evaluated = 0
        self.spin = np.zeros((*([self.mesh_points] * self.model.dim), 3))

    def populate_spin(self, filled_bands=None):
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        if (self.spin_evaluated and self.spin_evaluated == filled_bands) or self.S is None:
            return
        if not self.evaluated:
            self.populate_mesh()
        self.spin = np.tensordot(self.S, np.conj(self.states[..., :filled_bands]) @ np.swapaxes(self.states[..., :filled_bands], -1, -2), ([1, self.model.dim], [2, self.model.dim + 1])).transpose((1, 2, 0)).real
        self.spin_evaluted = filled_bands

    def minimum_spin_gap(self, filled_bands=None):
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        self.populate_spin(filled_bands)
        return np.min(np.sqrt(np.sum(self.spin**2, axis=2)))

    def normalized_spin(self):
        s = np.sqrt(np.sum(self.spin**2, axis=2).reshape((self.mesh_points, self.mesh_points, 1)))
        s[s == 0] = np.finfo(np.float32).eps
        return self.spin / s

    def compute_skyrmion(self, filled_bands=None):
        """S has shape (3, bands, bands).
        """
        if self.S is None:
            return
        if self.model.dim != 2:
            print("Computation of skyrmion number is only supported in 2-D.")
            return
        self.populate_spin(filled_bands)

        # Normalize the spin for the computation
        expt = np.concatenate((np.zeros((self.mesh_points, self.mesh_points, 1)), self.normalized_spin()), axis=2)
        
        lat = lambda k: np.round((np.array(k) + np.pi) / (2 * np.pi / (self.mesh_points - 1))).astype(int)
        model = TwoBandModel(d=lambda k: expt[tuple(lat(k))])
        sim = Simulator(model, self.mesh_points)
        return sim.compute_chern()

    def compute_skyrmion_z2(self, Ss, filled_bands=None, SOC=True):
        if self.model.dim != 2:
            print("Computation of skyrmion Z2 invariant is only supported in 2-D.")
            return
        if not SOC:
            if filled_bands and filled_bands % 2 == 1:
                print("Error: filled_bands cannot be odd in Z2 computation!")
                return -1
            model_s1 = Model()
            model_s2 = Model()
            model_s1.hamiltonian = lambda k: self.model.hamiltonian(k)[:self.model.bands // 2, :self.model.bands // 2]
            model_s2.hamiltonian = lambda k: self.model.hamiltonian(k)[self.model.bands // 2:, self.model.bands // 2:]
            model_s1.dim = model_s2.dim = self.model.dim
            model_s1.bands = model_s2.bands = self.model.bands // 2
            sim_s1 = Simulator(model_s1, self.mesh_points)
            sim_s2 = Simulator(model_s2, self.mesh_points)
            sim_s1.set_spin_op(Ss)
            sim_s2.set_spin_op(Ss)
            v = ((sim_s1.compute_skyrmion(filled_bands // 2) - sim_s2.compute_skyrmion(filled_bands // 2)) / 2) % 2
            del model_s1, model_s2, sim_s1, sim_s2
            return v
        else:
            raise NotImplementedError

    def plot_spin_texture(self, filled_bands=None, normalize=True):
        if self.model.dim != 2:
            print("Spin texture plotting is only supported in 2-D.")
            return

        self.populate_spin(filled_bands)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        kx, ky, kz = np.meshgrid(self.mesh, self.mesh, np.zeros(1))

        s = self.spin if not normalize else self.normalized_spin()
        ax.quiver(kx, ky, kz, s[:, :, 0:1], s[:, :, 1:2], s[:, :, 2:3], length=2 * np.pi / self.mesh_points, arrow_length_ratio=0.2)
        ax.set_zlim(-np.pi, np.pi)
        plt.show()

    def plot_spin_heat_map(self, filled_bands=None, normalize=True):
        if self.model.dim != 2:
            print("Spin texture plotting is only supported in 2-D.")
            return

        self.populate_spin(filled_bands)
        s = self.spin if not normalize else self.normalized_spin()

        for i in range(3):
            fig = plt.figure()
            ax = fig.gca()
            pos = ax.imshow(s[:, :, i], cmap="coolwarm", extent=(-np.pi, np.pi, -np.pi, np.pi))
            fig.colorbar(pos, ax=ax)
            plt.show()
