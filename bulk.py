import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define Pauli matrices for later use
pauli = [0] * 4
pauli[0] = np.eye(2)
pauli[1] = np.array([[0, 1], [1, 0]])
pauli[2] = np.array([[0, -1j], [1j, 0]])
pauli[3] = np.array([[1, 0], [0, -1]])
pauli = np.array(pauli)

# Define free parameters
u = 1.0

# Define generic base class for models
class Model:
    name = NotImplemented       # The name of the model
    dim = NotImplemented        # The dimension of the model
    bands = NotImplemented      # The number of bands of the model

    def __init__(self, **parameters):
        self.parameters = parameters

    def hamiltonian(self, k):
        raise NotImplementedError
    
    def get_parameters(self):
        return self.parameters

class QWZModel(Model):
    """Models the QWZ Hamiltonian H = d . sigma,
    where sigma are the Pauli matrices and d = (sin(kx), sin(ky), u + cos(kx) + cos(ky)).
    """
    name = "QWZ"
    dim = 2
    bands = 2

    def __init__(self, **parameters):
        super().__init__(**parameters)
        
        if "u" not in self.parameters:
            print("Required parameter u not found.")
            self.u = 0
        else:
            self.u = float(self.parameters["u"])

        for key in self.parameters:
            if key != "u":
                print("Parameter", key, "provided, but is not used by the QWZ model.")


    def hamiltonian(self, k):
        k = np.asarray(k, dtype=np.float64).flatten()
        assert k.shape == (self.dim,)
        d = np.concatenate((np.sin(k), np.array([np.sum(np.cos(k)) + self.u])))
        return np.tensordot(d, pauli[1:, :, :], (0, 0))

class BHZModel(Model):
    name = "BHZ"
    dim = 2
    bands = 4

    def __init__(self, **parameters):
        super().__init__(**parameters)

        if "u" not in self.parameters:
            print("Required parameter u not found.")
            self.u = 0
        else:
            self.u = float(self.parameters["u"])

        if "SOC" in self.parameters:
            self.C = self.parameters["SOC"]
        else:
            self.C = np.zeros(2)

        for key in self.parameters:
            if key not in ["u", "SOC"]:
                print("Parameter", key, "provided, but is not used by the BHZ model.")

    def hamiltonian(self, k):
        k = np.asarray(k, dtype=np.float64).flatten()
        assert k.shape == (self.dim,)
        d = np.concatenate((np.sin(k), np.array([np.sum(np.cos(k)) + self.u])))
        return np.kron(sigma[0], d[3] * sigma[3] + d[2] * sigma[2]) + np.kron(sigma[3], d[1] * sigma[1]) + np.kron(sigma[1], self.C)


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

    def plot_band(self, filled_bands=None):
        if not self.evaluated:
            self.populate_mesh()
        if not filled_bands:
            filled_bands = self.model.bands // 2    # Default to half-filling

        if self.model.dim == 1:
            raise NotImplementedError
        elif self.model.dim == 2:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            X, Y = np.meshgrid(self.mesh, self.mesh)
            ax.plot_surface(X, Y, self.band[:, :, filled_bands - 1])
            ax.plot_surface(X, Y, self.band[:, :, filled_bands])
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
        for i in range(filled_bands):
            F = np.sum(np.conj(self.states[:-1, :-1, :, i]) * self.states[1:, :-1, :, i], axis=2)
            F *= np.sum(np.conj(self.states[1:, :-1, :, i]) * self.states[1:, 1:, :, i], axis=2)
            F *= np.sum(np.conj(self.states[1:, 1:, :, i]) * self.states[:-1, 1:, :, i], axis=2)
            F *= np.sum(np.conj(self.states[:-1, 1:, :, i]) * self.states[:-1, :-1, :, i], axis=2)
            F = -np.angle(F)
            Q += np.sum(F) / 2.0 / np.pi
        return Q

    def compute_z2(self, filled_bands=None, SOC=False):
        if self.model.dim != 2:
            print("Computation of Z2 invariant is only supported in d=2.")
            return

        if not SOC:
            model_s1 = Model()
            model_s2 = Model()
            model_s1.hamiltonian = lambda k: self.model.hamiltonian(k)[:self.model.bands // 2, :self.model.bands // 2]
            model_s2.hamiltonian = lambda k: self.model.hamiltonian(k)[self.model.bands // 2:, self.model.bands // 2:]
            sim_s1 = Simulation(model_s1, self.mesh_points)
            sim_s2 = Simulation(model_s2, self.mesh_points)
            if filled_bands and filled_bands % 2 == 1:
                print("Error: filled_bands cannot be odd in Z2 computation!")
                return -1
            v = ((sim_s1.compute_chern(filled_bands // 2 if filled_bands else None) - sim_s2.compute_chern(filled_bands // 2 if filled_bands else None)) // 2) % 2
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
            for i in range(filled_bands):
                F = np.sum(np.conj(self.states[n:-1, :-1, :, i]) * self.states[n + 1:, :-1, :, i], axis=2)
                F *= np.sum(np.conj(self.states[n + 1:, :-1, :, i]) * self.states[1:, 1:, :, i], axis=2)
                F *= np.sum(np.conj(self.states[n + 1:, 1:, :, i]) * self.states[n:-1, 1:, :, i], axis=2)
                F *= np.sum(np.conj(self.states[n:-1, 1:, :, i]) * self.states[n:-1, :-1, :, i], axis=2)
                F = -np.angle(F)
                Q += np.sum(F)

            # Compute integral of Berry connection about edge of effective Brillouin zone
            raise NotImplementedError

    def wilson_loop(self, loop, points=100, filled_bands=None, phases=False):
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        W = np.eye(self.model.bands)
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
# for i, u in zip(range(51), np.linspace(-3, 3, 51)):
#     sim = Simulator(QWZModel(u=u), 21)
#     # print("u = %d, BG = %.2f" % (i, sim.direct_band_gap()))
#     # sim.plot_band()
#     # print(sim.compute_chern())
#     chern[i] = sim.compute_chern()
#     del sim
# plt.plot(np.linspace(-3, 3, 51), chern, "o-")
# plt.show()

phases = np.zeros(100)
for i, ky in zip(range(100), np.linspace(-np.pi, np.pi, 100)):
    sim = Simulator(QWZModel(u=-1.5), 21)
    p = sim.wilson_loop(lambda x: (2*np.pi*(x - 1/2), ky), phases=True)
    phases[i] = p[0]
# print(phases)
plt.plot(np.linspace(-np.pi, np.pi, 100), phases)
plt.ylim(-np.pi, np.pi)
plt.show()

# for i in range(-3, 4):
#     sim = Simulator(BHZModel(u=i), 21)
#     print("u = %d, BG = %.2f: % (i, sim.direct_band_gap()))
