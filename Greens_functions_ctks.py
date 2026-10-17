from utils import conjugate
import numpy as np
from numpy import linalg 
from pytreenet.ttns.ttns_ttno.application import (
    apply_ttno_to_ttns,
    ApplicationMethod,
)
from pytreenet.time_evolution.bug import BUGConfig
from pytreenet.time_evolution.time_evolution import TimeEvoMode
from scipy import sparse
from scipy.linalg import expm
from pytreenet.time_evolution.bug import BUG, BUGConfig
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.operators import TensorProduct
import copy
from copy import deepcopy
from utils import construct_spinful_fermionic_operators
from CTKS_BUG import BUG_time_evolution
from numpy.linalg import inv
import scipy.integrate as integrate
from scipy.integrate import quad_vec
from copy import deepcopy
from scipy.integrate import simpson

def Greens_function_lehman(
    c,
    E,
    psi,
    n,
    w,
    E_0=None,
    psi_0=None,
):
    """
    Calculate a zero-temperature retarded Green's function from a
    complete or truncated set of eigenpairs.

    Parameters
    ----------
    c
        Fermionic annihilation operator in the full Hilbert space.

    E : array_like, shape (n_ritz,)
        Exact energies or truncated Ritz energies.

    psi : array_like, shape (hilbert_dim, n_ritz)
        Corresponding eigenvectors or Ritz vectors as columns.
        For truncated Krylov results, these must first be transformed
        back into the full Hilbert space.

    n : float
        Positive broadening parameter eta.

    w : float or array_like
        Frequency or frequency grid.

    E_0 : float, optional
        Ground-state energy. If omitted, the lowest supplied energy
        is used.

    psi_0 : array_like, optional
        Ground-state vector in the full Hilbert space. If omitted,
        the vector corresponding to the lowest supplied energy is used.

    Returns
    -------
    G
        Approximate retarded Green's function.
    """
    E = np.asarray(E)
    psi = np.asarray(psi, dtype=np.complex128)
    w = np.asarray(w, dtype=np.float64)

    if n <= 0.0:
        raise ValueError("The broadening parameter n must be positive.")

    # Allow only negligible imaginary parts in the Ritz energies.
    E = np.real_if_close(E)

    if np.iscomplexobj(E):
        raise ValueError(
            "The supplied energies contain significant imaginary parts."
        )

    E = np.asarray(E, dtype=np.float64).reshape(-1)

    if psi.ndim != 2:
        raise ValueError(
            "psi must have shape (Hilbert-space dimension, number of states)."
        )

    if psi.shape[1] != len(E):
        raise ValueError(
            f"psi contains {psi.shape[1]} states, but len(E)={len(E)}."
        )

    if c.shape[0] != psi.shape[0]:
        raise ValueError(
            "The Ritz vectors are not represented in the full Hilbert "
            "space. Reconstruct them from the Krylov coefficients first."
        )

    # Sort the supplied exact or Ritz eigenpairs.
    order = np.argsort(E)
    E = E[order]
    psi = psi[:, order]

    # If no external ground state is given, use the lowest supplied state.
    if psi_0 is None:
        psi_0 = psi[:, 0]
    else:
        psi_0 = np.asarray(
            psi_0,
            dtype=np.complex128,
        ).reshape(-1)

    if psi_0.shape[0] != psi.shape[0]:
        raise ValueError(
            "psi_0 and the Ritz vectors must belong to the same "
            "Hilbert space."
        )

    norm_0 = np.linalg.norm(psi_0)

    if norm_0 == 0.0:
        raise ValueError("psi_0 has zero norm.")

    psi_0 = psi_0 / norm_0

    if E_0 is None:
        E_0 = E[0]
    else:
        E_0 = float(np.real(E_0))

    c_dag = c.conj().T

    particle_state = c_dag @ psi_0
    hole_state = c @ psi_0

    particle_amplitudes = psi.conj().T @ particle_state
    hole_amplitudes = psi.conj().T @ hole_state

    particle_weights = np.abs(particle_amplitudes) ** 2
    hole_weights = np.abs(hole_amplitudes) ** 2

    excitation_energies = E - E_0
    z = w[..., None] + 1.0j * n

    particle_part = np.sum(
        particle_weights
        / (z - excitation_energies),
        axis=-1,
    )

    hole_part = np.sum(
        hole_weights
        / (z + excitation_energies),
        axis=-1,
    )

    G = particle_part + hole_part

    if w.ndim == 0:
        return complex(G)

    return G

def spectral_function(G):
    """Calculate A(w) = -Im[G^R(w)] / pi."""
    return -np.imag(np.asarray(G)) / np.pi



def Greens_function_lehman_TTN(E,Ritz_coefficients,Krylo_vectors,psi_0,E_0,eta,w,node_key=None):
    """Evaluates the singe particle Green's function in the Lehmann representation"""
    #A = ApplicationMethod("direct")
    idx = np.argsort(E)
    E=E[idx]
    if node_key==None:
        dict_keys=list(Krylo_vectors[0].nodes.keys())
        node_key=dict_keys[0]
    d = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ],
    dtype=np.complex128)
    #d_dag= conjugate(d)
    operator = TensorProduct({node_key:d})
    m=len(Krylo_vectors)
    C= np.zeros((m,m),dtype=np.complex128)

    C_ttn= copy.deepcopy(Krylo_vectors)
    for i in range(m):
        C_ttn[i].apply_operator(operator)

    for i in range(m):
        for j in range(m):
            C[i][j]= Krylo_vectors[j].scalar_product(C_ttn[i])


    C_dagg= conjugate(C)

    #E_0= E[0]
    #psi_0= Ritz_coefficients[:,0]
    total_n= len(E)
    term_1=0
    term_2=0
    G=0.0+0.0j
    for i in range(total_n):
        term_1+= (np.vdot(psi_0,C @ Ritz_coefficients[:,i])*np.vdot(Ritz_coefficients[:,i],C_dagg @ psi_0))/(w+1.0j*eta+E_0-E[i])
        term_2+= (np.vdot(psi_0,C_dagg @ Ritz_coefficients[:,i])*np.vdot(Ritz_coefficients[:,i],C @ psi_0))/(w+1.0j*eta+E[i]-E_0)
    G = term_1+term_2
    return G



def lehmann_sector_TTN(
    E, Ritz_coefficients, Krylov_vectors, psi_exc,
    E_0, eta, omega, sign,
):
    """sign=+1: addition; sign=-1: removal."""
    E = np.asarray(E)
    B = np.asarray(Ritz_coefficients)  # B = X @ Q; shape (n_vectors, n_ritz)

    order = np.argsort(E)
    E, B = E[order], B[:, order]

    # PyTreeNet convention: v.scalar_product(psi_exc) = <psi_exc|v>
    overlaps = np.array(
        [v.scalar_product(psi_exc) for v in Krylov_vectors],
        dtype=complex,
    )
    weights = np.abs(overlaps @ B) ** 2

    z = np.asarray(omega)[..., None] + 1j * eta
    G = np.sum(weights / (z - sign * (E - E_0)), axis=-1)
    return G, weights.sum()



def _bug_step(
    state,
    duration,
    TTNO,
    operators,
    max_bond_dim,
    time_step_size=0.1,
    rel_tol=1e-12,
    abs_tol=1e-12,
):
    """Evolve a TTNS by exp(-i H duration) for duration >= 0."""

    duration = float(duration)

    if duration < 0.0:
        raise ValueError("BUG evolution requires a non-negative duration.")

    if duration == 0.0:
        return state

    tree_bug, *_ = BUG_time_evolution(
        Tree_state=state,
        TTNO_Hamil=TTNO,
        time_step_size=min(time_step_size, duration),
        final_time=duration,
        operators=operators,
        max_bond_dim=max_bond_dim,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )

    return tree_bug.state


def real_time_evolution_TTN(
    t_end,
    steps,
    psi_ket,
    bra_vectors,
    TTNO,
    operators,
    max_bond_dim,
    eta,
    E_0=0.0,
    time_step_size=0.1,
):
    r"""
    Calculate

        C_i(t) = exp(i E_0 t) <bra_i|exp(-i H t)|psi_ket>

    on a uniform time grid.

    `steps` denotes the number of time intervals, so the returned arrays
    contain `steps + 1` points, including t=0.
    """

    if t_end <= 0.0:
        raise ValueError("t_end must be positive.")

    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer.")

    bra_vectors = list(bra_vectors)
    time_grid = np.linspace(0.0, t_end, steps + 1)

    correlation = np.empty(
        (len(time_grid), len(bra_vectors)),
        dtype=np.complex128,
    )

    # Preserve the original input state.
    psi_t = deepcopy(psi_ket)
    previous_time = 0.0

    for n, t in enumerate(time_grid):
        delta_t = t - previous_time

        psi_t = _bug_step(
            state=psi_t,
            duration=delta_t,
            TTNO=TTNO,
            operators=operators,
            max_bond_dim=max_bond_dim,
            time_step_size=time_step_size,
        )

        phase = np.exp(1j * (E_0+1j*eta) * t)

        correlation[n, :] = phase * np.asarray(
            [
                psi_t.scalar_product(bra)
                for bra in bra_vectors
            ],
            dtype=np.complex128,
        )

        previous_time = t

    return time_grid, correlation

def Greens_function_CTKS_time_domain(
    psi_A,
    psi_B,
    Vectors,
    Q,
    X,
    E,
    t_end,
    steps,
    interval,
    TTNO_minus_H,
    operators,
    max_bond_dim,
    eta,
    E_0=0.0,
    time_step_size=0.1,
):
    r"""
    CTKS approximation to

        C_AB(t) = exp(i E_0 t) <psi_A|exp(-i H t)|psi_B>.

    Parameters
    ----------
    interval:
        CTKS construction interval T.

    TTNO_minus_H:
        TTNO representing -H. It permits bra-side propagation using only
        positive BUG time steps.

    E:
        Absolute Ritz energies of the projected Hamiltonian.
    """

    if t_end <= 0.0:
        raise ValueError("t_end must be positive.")

    if interval <= 0.0:
        raise ValueError("interval must be positive.")

    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer.")

    Vectors = list(Vectors)
    X = np.asarray(X, dtype=np.complex128)
    Q = np.asarray(Q, dtype=np.complex128)
    E = np.asarray(E, dtype=float)

    number_of_vectors = len(Vectors)

    if X.ndim != 2 or X.shape[0] != number_of_vectors:
        raise ValueError(
            "X must be a matrix with X.shape[0] == len(Vectors)."
        )

    if Q.ndim != 2 or Q.shape[0] != X.shape[1]:
        raise ValueError(
            "Q must be a matrix with Q.shape[0] == X.shape[1]."
        )

    if Q.shape[1] != len(E):
        raise ValueError("Q.shape[1] must equal len(E).")

    time_grid = np.linspace(0.0, t_end, steps + 1)

    # Decompose t = pT + tau.
    interval_number = np.floor(time_grid / interval).astype(int)
    remainder = time_grid - interval_number * interval

    tolerance = (
        32.0
        * np.finfo(float).eps
        * max(1.0, t_end, interval)
    )

    # Correct floating-point errors near interval boundaries.
    near_upper_boundary = np.abs(remainder - interval) < tolerance
    interval_number[near_upper_boundary] += 1
    remainder[near_upper_boundary] = 0.0
    remainder[np.abs(remainder) < tolerance] = 0.0

    # Transformation from the nonorthogonal CTKS vectors to Ritz states.
    basis_to_ritz = X @ Q

    # M_j = <v_j|psi_B>
    M = np.asarray(
        [
            psi_B.scalar_product(vector)
            for vector in Vectors
        ],
        dtype=np.complex128,
    )

    right_ritz = basis_to_ritz.conj().T @ M

    # Only the distinct remainder times have to be propagated.
    unique_remainders, remainder_indices = np.unique(
        remainder,
        return_inverse=True,
    )

    Y = np.empty(
        (len(unique_remainders), number_of_vectors),
        dtype=np.complex128,
    )

    # To calculate <psi_A|exp(-i H tau)|v_j>, evolve psi_A with -H:
    #
    # <exp(+i H tau) psi_A|v_j>
    #     = <psi_A|exp(-i H tau)|v_j>.
    psi_A_tau = deepcopy(psi_A)
    previous_tau = 0.0

    for n, tau in enumerate(unique_remainders):
        delta_tau = tau - previous_tau

        psi_A_tau = _bug_step(
            state=psi_A_tau,
            duration=delta_tau,
            TTNO=TTNO_minus_H,
            operators=operators,
            max_bond_dim=max_bond_dim,
            time_step_size=time_step_size,
        )

        Y[n, :] = np.exp(1j * (E_0+1j*eta) * tau) * np.asarray(
            [
                vector.scalar_product(psi_A_tau)
                for vector in Vectors
            ],
            dtype=np.complex128,
        )

        previous_tau = tau

    correlation = np.empty(len(time_grid), dtype=np.complex128)

    for n, number_of_intervals in enumerate(interval_number):
        complete_time = number_of_intervals * interval

        # Projected propagation through p complete CTKS intervals.
        ritz_phase = np.exp(
            -1j * (E - E_0-1j*eta) * complete_time
        )

        coefficients = basis_to_ritz @ (
            ritz_phase * right_ritz
        )

        # Ordinary dot product: Y already contains matrix elements and
        # must not be conjugated again.
        correlation[n] = np.dot(
            Y[remainder_indices[n]],
            coefficients,
        )

    return time_grid, correlation


def boost_fac(X,Q,E,E_0,T):
    U= np.exp(-1.0j*(E-E_0)*T)/(1-np.exp(-1.0j*(E-E_0)*T))
    D= X @ Q @ U @ Q.conjugate() @ X.conjugate() 
    return D

def Boost_op(S,X,Q,E,E_0,T,Y):
    M= S[:,0]
    D= boost_fac(X,Q,E,E_0,T)
    K= Y@ D @ M
    return K





def ctks_time_parts(
    psi_A, psi_B, Vectors, Q, X, E,
    t_end, steps, interval, TTNO_minus_H, operators, max_bond_dim,
    eta=0.0, E_0=0.0, time_step_size=0.1, *, bug_step,
):
    r"""Return (time, C_real, K_time, C_total) as complex-valued time series.

    C_total approximates exp((i*E_0 - eta)*t) <psi_A|exp(-i*H*t)|psi_B>.
    C_real contains the direct correlation for 0 <= t < interval, and zeros
    afterward. K_time contains the CTKS-extrapolated tail for t >= interval,
    and zeros before it. Thus C_total = C_real + K_time. K_time is the
    time-domain tail whose Fourier transform gives the long-time correction;
    it is not the frequency-dependent geometric boost evaluated at time t.

    H must be Hermitian; TTNO_minus_H represents -H. E contains absolute Ritz
    energies, X has shape (number of vectors, orthonormal rank), and Q has
    shape (orthonormal rank, number of retained Ritz states).

    This uses the PyTreeNet convention ket.scalar_product(bra) = <bra|ket>.
    Pass your existing propagation helper as bug_step=_bug_step. It must
    evolve for the requested duration without normalizing the state.

    interval must be an integer multiple of dt = t_end / steps. Only one
    bra-side trajectory up to interval - dt is propagated. The original
    states are preserved. The retarded prefactor -i and sector weights are
    not included. Damping is already applied; do not apply eta a second time.
    """
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError("t_end must be finite and positive.")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if not np.isfinite(interval) or not 0 < interval <= t_end:
        raise ValueError("interval must satisfy 0 < interval <= t_end.")
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative.")
    if not np.isfinite(E_0) or not np.isfinite(time_step_size) or time_step_size <= 0:
        raise ValueError("E_0 must be finite and time_step_size positive.")

    dt = t_end / steps
    ratio = interval / dt
    n_interval = int(round(ratio))
    tolerance = max(1e-10, 64 * np.finfo(float).eps * abs(ratio))
    if n_interval < 1 or abs(ratio - n_interval) > tolerance:
        raise ValueError("Choose interval as an integer multiple of t_end / steps.")
    # Align the interval exactly with the returned uniform grid.
    T = n_interval * dt

    Vectors = list(Vectors)
    X = np.asarray(X, dtype=complex)
    Q = np.asarray(Q, dtype=complex)
    E = np.real_if_close(np.asarray(E))
    if np.iscomplexobj(E) or E.ndim != 1 or not np.all(np.isfinite(E)):
        raise ValueError("E must be a finite, real, one-dimensional energy array.")
    E = E.astype(float)
    if not Vectors or X.ndim != 2 or X.shape[0] != len(Vectors):
        raise ValueError("X.shape[0] must match a nonempty Vectors list.")
    if Q.ndim != 2 or Q.shape != (X.shape[1], len(E)) or len(E) == 0:
        raise ValueError("Q must have shape (X.shape[1], len(E)), with len(E) > 0.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Q)):
        raise ValueError("X and Q must have finite entries.")

    time = np.linspace(0.0, t_end, steps + 1)
    C_real = np.zeros(steps + 1, dtype=complex)
    K_time = np.zeros(steps + 1, dtype=complex)

    R = X @ Q
    M = np.asarray([psi_B.scalar_product(v) for v in Vectors], dtype=complex)
    right_ritz = R.conj().T @ M
    Y_ritz = np.empty((n_interval, len(E)), dtype=complex)

    # Evolving psi_A with -H yields <psi_A|exp(-i H tau) on the bra side.
    bra_state = deepcopy(psi_A)
    for j in range(n_interval):
        if j > 0:
            bra_state = bug_step(
                state=bra_state, duration=dt, TTNO=TTNO_minus_H,
                operators=operators, max_bond_dim=max_bond_dim,
                time_step_size=time_step_size,
            )
        tau = j * dt
        phase = np.exp((1j * E_0 - eta) * tau)
        # Preserve direct short-time information, even if Ritz states were dropped.
        C_real[j] = phase * psi_B.scalar_product(bra_state)
        Y = np.asarray([v.scalar_product(bra_state) for v in Vectors], dtype=complex)
        Y_ritz[j] = phase * (Y @ R)

    # Integer arithmetic reuses each remainder time exactly; no np.unique needed.
    for n in range(n_interval, steps + 1):
        p, j = divmod(n, n_interval)
        phases = np.exp((-1j * (E - E_0) - eta) * (p * T))
        K_time[n] = Y_ritz[j] @ (phases * right_ritz)

    return time, C_real, K_time, C_real + K_time

