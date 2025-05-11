from flask import Flask, request, jsonify
import numpy as np
from scipy.linalg import eigh_tridiagonal

app = Flask(__name__)

def solve_schrodinger(V_func, x_min=-5, x_max=5, N=1000, mass=1.0, hbar=1.0, num_states=5):
    # Spatial grid
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    
    # Potential
    V = V_func(x)
    
    # Kinetic energy matrix (tridiagonal)
    factor = hbar**2 / (2 * mass * dx**2)
    diagonal = 2 * factor + V
    off_diagonal = -factor * np.ones(N - 1)
    
    # Solve eigenvalue problem
    energies, wavefuncs = eigh_tridiagonal(diagonal, off_diagonal)
    
    # Normalize wavefunctions
    norm_wavefuncs = []
    for n in range(num_states):
        psi = wavefuncs[:, n]
        psi /= np.sqrt(np.trapz(psi**2, x))  # Normalize
        norm_wavefuncs.append(psi)
    
    return x, V, energies[:num_states], norm_wavefuncs

@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    potential_str = data['potential']  # user input potential function
    # Convert the potential string to a valid function
    def V_func(x):
        return eval(potential_str)  # Be cautious with eval; for now, assume safe input.
    
    x, V, energies, wavefuncs = solve_schrodinger(V_func)

    # Return results as JSON
    return jsonify({
        'x': x.tolist(),
        'V': V.tolist(),
        'energies': energies.tolist(),
        'wavefuncs': [psi.tolist() for psi in wavefuncs]
    })

if __name__ == '__main__':
    app.run(debug=True)
