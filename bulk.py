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
            print("Computation of Chern numbers is only supported in d=2.")
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

    def compute_skyrmion(self, S, filled_bands=None):
        """S has shape (3, bands, bands).
        """
        if not self.evaluated:
            self.populate_mesh()
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        expt = np.tensordot(S, np.conj(self.states[:, :, :, :filled_bands]) @ self.states[:, :, :, :filled_bands].transpose((0, 1, 3, 2)), ([1, 2], [2, 3]))
        expt = np.vstack((np.zeros((1, self.mesh_points, self.mesh_points)), expt)).transpose((1, 2, 0))
        
        lat = lambda k: ((np.array(k) + np.pi) // (2 * np.pi / self.mesh_points)).astype(int)
        model = TwoBandModel(d=lambda k: expt[tuple(lat(k))])
        sim = Simulator(model, self.mesh_points)
        return sim.compute_chern()

    def compute_z2(self, filled_bands=None, SOC=False):
        if self.model.dim != 2:
            print("Computation of Z2 invariant is only supported in d=2.")
            return

        if not filled_bands:
            filled_bands = self.model.bands // 2

        if not SOC:
            model_s1 = Model()
            model_s2 = Model()
            model_s1.hamiltonian = lambda k: self.model.hamiltonian(k)[:self.model.bands // 2, :self.model.bands // 2]
            model_s2.hamiltonian = lambda k: self.model.hamiltonian(k)[self.model.bands // 2:, self.model.bands // 2:]
            model_s1.dim = self.model.dim
            model_s2.dim = self.model.dim
            model_s1.bands = self.model.bands // 2
            model_s2.bands = self.model.bands // 2
            sim_s1 = Simulator(model_s1, self.mesh_points)
            sim_s2 = Simulator(model_s2, self.mesh_points)
            if filled_bands and filled_bands % 2 == 1:
                print("Error: filled_bands cannot be odd in Z2 computation!")
                return -1
            v = ((sim_s1.compute_chern(filled_bands // 2) - sim_s2.compute_chern(filled_bands // 2)) / 2) % 2
            del model_s1, model_s2, sim_s1, sim_s2
            return v
        else:
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
            W = np.angle(scipy.linalg.det(W))
            
            return ((W - Q) / 2 / np.pi) % 2


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
        return W if not phases else np.sort(np.angle(scipy.linalg.eig(self.wilson_loop(loop, points, filled_bands))[0]))

# chern = np.zeros(51)
# skyr = np.zeros(51)
# for i, u in zip(range(51), np.linspace(-3, 3, 51)):
#     sim = Simulator(FourBandModel(b=1.0, k=u), 21)
#     # print("u = %d, BG = %.2f" % (i, sim.direct_band_gap()))
#     # sim.plot_band()
#     # print(sim.compute_chern())
#     chern[i] = sim.compute_chern()
#     skyr[i] = sim.compute_skyrmion((np.kron(pauli[3], pauli[1]), np.kron(pauli[0], pauli[2]), np.kron(pauli[3], pauli[3])))
#     del sim
# plt.plot(np.linspace(-3, 3, 51), chern, "o-")
# plt.title("Chern number of four band model")
# plt.xlabel("k")
# plt.ylabel("C")
# plt.show()
# plt.plot(np.linspace(-3, 3, 51), skyr, "o-")
# plt.title("Skyrmion number of four band model")
# plt.xlabel("u")
# plt.ylabel("Q")
# plt.show()

# phases = np.zeros((2, 100))
# for i, ky in zip(range(100), np.linspace(-np.pi, np.pi, 100)):
#     sim = Simulator(BHZModel(u=0.6), 21)
#     p = sim.wilson_loop(lambda x: (2*np.pi*(x - 1/2), ky), phases=True)
#     phases[:, i] = p
# # print(phases)
# plt.plot(np.linspace(-np.pi, np.pi, 100), phases[0], "o-")
# plt.plot(np.linspace(-np.pi, np.pi, 100), phases[1], "o-")
# plt.ylim(-np.pi, np.pi)
# plt.show()

# z2 = np.zeros(50)
# for i, u in zip(range(50), np.linspace(-3, 3, 50)):
#     sim = Simulator(BHZModel(u=u), 21)
#     # print("u = %.2f, BG = %.2f" % (i, sim.direct_band_gap()))
#     z2[i] = sim.compute_z2(SOC=True)
#     print("u = %.1f, Z2 = %.3f" % (u, z2[i]))
#     del sim
# plt.plot(np.linspace(-3, 3, 50), np.round(z2) % 2)
# plt.show()
