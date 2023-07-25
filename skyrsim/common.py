import numpy as np

# Define Pauli matrices for later use
pauli = np.zeros((4, 2, 2), dtype=np.complex64)
pauli[0] = np.eye(2)
pauli[1] = np.array([[0, 1], [1, 0]])
pauli[2] = np.array([[0, -1j], [1j, 0]])
pauli[3] = np.array([[1, 0], [0, -1]])

# Define Gell-Mann matrices
gellmann = np.zeros((9, 3, 3), dtype=np.complex64)
gellmann[0] = np.eye(3)
gellmann[1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
gellmann[2] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
gellmann[3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
gellmann[4] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
gellmann[5] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
gellmann[6] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
gellmann[7] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
gellmann[8] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3)
