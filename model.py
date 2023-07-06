import numpy as np
import scipy

# Define Pauli matrices for later use
pauli = [0] * 4
pauli[0] = np.eye(2)
pauli[1] = np.array([[0, 1], [1, 0]])
pauli[2] = np.array([[0, -1j], [1j, 0]])
pauli[3] = np.array([[1, 0], [0, -1]])
pauli = np.array(pauli)

# Define generic base class for models
class Model:
    name = NotImplemented       # The name of the model
    dim = NotImplemented        # The dimension of the model
    bands = NotImplemented      # The number of bands of the model
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

    def open_hamiltonian(self, N, k):
        """Opens boundary conditions assuming tight-binding, i.e., no direct interactions between the first
        and the last cell along the dimensions opened.
        Assumes square lattice and a lattice parameter of 1.
        """
        open_dim = np.arange(len(N))[np.array(N) > 0]
        fft = None
        if len(open_dim) == 0:
            return self.hamiltonian(k)
        elif len(open_dim) == 1:
            fft = lambda v: scipy.fft.fft(v, axis=0)
        else:
            fft = lambda v: scipy.fft.fft(v, axes=(0, 1))
            # TODO: implement this using recursion
            raise NotImplementedError
        
        ks = np.tile(k, (N[open_dim[0]], 1))
        ks[:, open_dim[0]] = np.linspace(0, 2 * np.pi, N[open_dim[0]], endpoint=False)
        v = np.array([self.hamiltonian(k) for k in ks])
        v = fft(v)
        v = np.hstack(list(v))

        hamil = np.zeros((np.prod(N[open_dim]) * self.bands, np.prod(N[open_dim]) * self.bands), dtype=np.complex64)

        for i in range(N[open_dim[0]]):
            hamil[i * self.bands:(i + 1) * self.bands] = np.roll(v, i * self.bands, axis=1)
        
        # Enforce open boundary conditions / no coupling between ends
        hamil[:self.bands, -self.bands:] = hamil[-self.bands:, :self.bands] = np.zeros((self.bands, self.bands))

        return hamil

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
        
