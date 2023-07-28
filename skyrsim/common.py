import numpy as np

# Define Pauli matrices
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

# Define spin matrices
def spin_matrices(s):
    if int(s) == s or int(s * 2) == s * 2:
        N = int(2 * s + 1)
        S = np.zeros((3, N, N), dtype=np.complex64)
        m = np.arange(N - 1) - s
        d = np.sqrt(s*(s+1) - m*(m+1)) / 2
        S[0] += np.diag(d, k=1)
        S[0] += np.diag(d, k=-1)
        S[1] += np.diag(d * -1j, k=1)
        S[1] += np.diag(d * 1j, k=-1)
        S[2] = np.diag(s - np.arange(N))
        return S
    raise ValueError("Only integer or half-integer spins accepted.")

