import numpy as np
import scipy
import scipy.linalg

from .common import *

# Define generic base class for models
class Model:
    name = NotImplemented       # The name of the model
    dim = NotImplemented        # The dimension of the model
    bands = NotImplemented      # The number of bands of the model
    hop_len = 1                 # Tight-binding hopping distance, default is 1 (nearest neighbor only)
    defaults = {}               # The parameters of the model and default values
    required = []               # The parameters required, the absence of which will produce a warning

    def __init__(self, **parameters):
        self.parameters = parameters

        for key in self.parameters:
            if key not in self.defaults:
                print("Parameter", key, "was provided, but is not used by the", self.name, "model.")

        for key in self.defaults:
            if key not in self.parameters:
                self.parameters[key] = self.defaults[key]
                if key in self.required:
                    print("Parameter", key, " is required by the", self.name, "model but is not provided.")

    def hamiltonian(self, k):
        raise NotImplementedError
    
    def get_parameters(self):
        return self.parameters

    def open_hamiltonian(self, N, k, PBC=None):
        """Opens boundary conditions assuming tight-binding, i.e., no direct interactions between the first
        and the last cell along the dimensions opened.
        Assumes square lattice and a lattice parameter of 1.
        """
        open_dim = np.arange(len(N))[np.array(N) > 0]
        fft = None
        hamiltonian = None
        sz = 0
        if len(open_dim) == 0:
            return self.hamiltonian(k)
        elif len(open_dim) == 1:
            fft = lambda v: scipy.fft.fft(v, axis=0)
            hamiltonian = self.hamiltonian
            sz = self.bands
        else:
            fft = lambda v: scipy.fft.fftn(v, axis=tuple(range(len(open_dim))))
            sN = np.array(N)
            sN[open_dim[0]] = 0
            hamiltonian = lambda k: self.open_hamiltonian(sN, k)
            sz = np.prod(N[open_dim[1:]]) * self.bands
        
        ks = np.tile(k, (N[open_dim[0]], 1))
        ks[:, open_dim[0]] = np.linspace(0, 2 * np.pi, N[open_dim[0]], endpoint=False)
        v = np.array([hamiltonian(k) for k in ks])
        v = fft(v) / N[open_dim[0]]
        v = np.hstack(list(v))

        hamil = np.zeros((N[open_dim[0]] * sz, N[open_dim[0]] * sz), dtype=np.complex64)

        for i in range(N[open_dim[0]]):
            hamil[i * sz:(i + 1) * sz] = np.roll(v, i * sz, axis=1)
        
        if PBC is None or not PBC[open_dim[0]]:
            # Enforce open boundary conditions / no coupling between ends
            for i in range(self.hop_len):
                hamil[i*sz:(i+1)*sz, -(self.hop_len-i)*sz:] = np.zeros((sz, (self.hop_len-i)*sz))
                hamil[-(self.hop_len-i)*sz:, i*sz:(i+1)*sz] = np.zeros(((self.hop_len-i)*sz, sz))

        return hamil

class Potential(Model):
    def __init__(self, model: Model, potential: callable, **parameters):
        self.model = model
        self.potential = potential

        self.name = self.model.name
        self.dim = self.model.dim
        self.bands = self.model.bands
        self.hop_len = self.model.hop_len

    def hamiltonian(self, k):
        return self.model.hamiltonian(k) + self.potential(k)

    def open_hamiltonian(self, N, k, PBC=None):
        return self.model.open_hamiltonian(N, k, PBC) + self.potential(N, k)


class TwoBandModel(Model):
    name = "Two Band"
    dim = 2
    bands = 2
    defaults = {"d": lambda k: np.zeros(4)}
    required = ["d"]

    def __init__(self, **parameters):
        super().__init__(**parameters)

    def hamiltonian(self, k):
        k = np.asarray(k, dtype=np.float64).flatten()
        assert k.shape == (self.dim,)
        return np.tensordot(self.parameters["d"](k), pauli[:, :, :], (0, 0))


class QWZModel(Model):
    """Models the QWZ Hamiltonian H = d . sigma,
    where sigma are the Pauli matrices and d = (sin(kx), sin(ky), u + cos(kx) + cos(ky)).
    """
    name = "QWZ"
    dim = 2
    bands = 2
    defaults = {"u": 0.0}
    required = ["u"]

    def __init__(self, **parameters):
        super().__init__(**parameters)

    def hamiltonian(self, k):
        k = np.asarray(k, dtype=np.float64).flatten()
        assert k.shape == (self.dim,)
        d = np.concatenate((np.sin(k), np.array([np.sum(np.cos(k)) + self.parameters["u"]])))
        return np.tensordot(d, pauli[1:, :, :], (0, 0))


class BHZModel(Model):
    name = "BHZ"
    dim = 2
    bands = 4
    defaults = {"u": 0.0, "SOC": np.zeros((2, 2))}
    required = ["u"]

    def __init__(self, **parameters):
        super().__init__(**parameters)

    def hamiltonian(self, k):
        k = np.asarray(k, dtype=np.float64).flatten()
        assert k.shape == (self.dim,)
        d = np.concatenate((np.sin(k), np.array([np.sum(np.cos(k)) + self.parameters["u"]])))
        hamil = np.kron(pauli[0], d[2] * pauli[3] + d[1] * pauli[2]) + np.kron(pauli[3], d[0] * pauli[1])
        hamil[-self.bands // 2:, :self.bands // 2] = self.parameters["SOC"]
        hamil[:self.bands // 2, -self.bands // 2:] = np.conj(self.parameters["SOC"]).T
        return hamil

class FourBandModel(Model):
    dim = 2
    bands = 4
    defaults = {"b": 1.0, "k": 1.0}
    required = ["b", "k"]

    def __init__(self, **parameters):
        super().__init__(**parameters)

    def hamiltonian(self, k):
        k = np.asarray(k, dtype=np.float64).flatten()
        assert k.shape == (self.dim,)
        return np.kron(pauli[3], (1 - np.sum(np.cos(k))) * pauli[3] + self.parameters["k"] * np.sin(k[1]) * pauli[1]) + np.kron(pauli[0], self.parameters["b"] * np.sin(k[0]) * pauli[2])
        
class TripletSticletModel(Model):
    name = "Sticlet Model"
    dim = 2
    bands = 4
    defaults = {"a": 0.0, "b": 0.0, "D0": 0.0}
    required = ["a", "b"]

    def hamiltonian(self, k):
        hamil = self.parameters["a"] * np.cos(k[0]) * np.kron(pauli[3], pauli[1])
        hamil += self.parameters["a"] * np.cos(k[1]) * np.kron(pauli[0], pauli[2])
        hamil += self.parameters["b"] * np.cos(k[0] + k[1]) * np.kron(pauli[3], pauli[3])
        hamil[:2, 2:] = 1j * self.parameters["D0"] * (np.sin(k[1]) * 1j * pauli[3] - np.sin(k[0]) * pauli[0])
        hamil[2:, :2] = hamil[:2, 2:].conj().T
        return hamil
