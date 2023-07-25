import pytest
from skyrsim.common import pauli, gellmann

def test_pauli_commutation():
    assert (pauli[1] @ pauli[2] - pauli[2] @ pauli[1]) == 2j * pauli[3]
    assert (pauli[2] @ pauli[3] - pauli[3] @ pauli[2]) == 2j * pauli[1]
    assert (pauli[3] @ pauli[1] - pauli[1] @ pauli[3]) == 2j * pauli[2]

def test_pauli_anticommutation():
    assert (pauli[1] @ pauli[1] + pauli[1] @ pauli[1]) == 2 * np.eye(2)
    assert (pauli[2] @ pauli[2] + pauli[2] @ pauli[2]) == 2 * np.eye(2)
    assert (pauli[3] @ pauli[3] + pauli[3] @ pauli[3]) == 2 * np.eye(2)
    assert (pauli[1] @ pauli[2] + pauli[2] @ pauli[1]) == np.zeros(2)
    assert (pauli[2] @ pauli[3] + pauli[3] @ pauli[2]) == np.zeros(2)
    assert (pauli[3] @ pauli[1] + pauli[1] @ pauli[3]) == np.zeros(2)
