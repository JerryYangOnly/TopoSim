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
                print("Parameter", key, "provided, but is not used by the model.")


    def hamiltonian(self, k):
        k = np.asarray(k, dtype=np.float64).flatten()
        assert k.shape == (self.dim,)
        d = np.concatenate((np.sin(k), np.array([np.sum(np.cos(k)) + self.u])))
        return np.tensordot(d, pauli[1:, :, :], (0, 0))


# A class for simulating results
class Simulator:
    def __init__(self, model: Model, mesh_points: int):
        self.model = model
        self.set_mesh(mesh_points)

    def set_mesh(self, mesh_points):
        self.mesh_points = mesh_points
        self.mesh = np.linspace(-np.pi, np.pi, mesh_points, endpoint=False)
        # self.meshX, self.meshY = np.meshgrid(np.linspace(-np.pi, np.pi, mesh_points, endpoint=False))
        self.evaluated = False
        self.band = np.zeros((mesh_points, mesh_points, self.model.bands))
        self.states = np.zeros((mesh_points, mesh_points, self.model.bands, self.model.bands), dtype=np.complex64)

    def populate_mesh(self):
        if self.evaluated:
            return False
        print("Starting evaluation")
        for i in tqdm(range(mesh_points)):
            for j in range(mesh_points):
                w, v = scipy.linalg.eigh(self.model.hamiltonian(self.mesh[i], self.mesh[j]))
                self.band[i, j, :] = w
                self.states[i, j, :, :] = v
        self.evaluated = True
        return True

    def direct_band_gap(self, filled_bands=None):
        if not self.evaluated:
            return -1
        if not filled_bands:
            filled_bands = self.model.bands // 2    # Default to half-filling
        return np.min(self.band[:, :, filled_bands] - self.band[:, :, filled_bands - 1])
