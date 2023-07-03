import numpy as np
import scipy
import matplotlib.pyplot as plt

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
    def __init__(self, **parameters):
        self.parameters = parameters

    def hamiltonian(self, k):
        raise NotImplementedError

class QWZModel(Model):
    """Models the QWZ Hamiltonian H = d . sigma,
    where sigma are the Pauli matrices and d = (sin(kx), sin(ky), u + cos(kx) + cos(ky)).
    """
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
        assert k.shape == (2,)
        d = np.concatenate((np.sin(k), np.array([np.sum(np.cos(k)) + self.u])))
        return np.tensordot(d, pauli[1:, :, :], (0, 0))


# A class for simulating results
class Simulator:
    def __init__(self, model: Model):
        self.model = model
