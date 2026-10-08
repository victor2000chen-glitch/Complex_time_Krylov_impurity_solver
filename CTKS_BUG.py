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
from numpy.linalg import inv
import scipy.integrate as integrate
from scipy.integrate import quad_vec
from copy import deepcopy
from scipy.integrate import simpson

#Time evolution of TTNS
def BUG_time_evolution(Tree_state,TTNO_Hamil,time_step_size,final_time,operators,max_bond_dim=20, rel_tol = 1e-16, abs_tol = 1e-16):
    """Time evolution of TTN using the bug integrator"""
    bug_config = BUGConfig(
        max_bond_dim=max_bond_dim,
        rel_tol=rel_tol,
        total_tol=abs_tol,
        time_evo_mode=TimeEvoMode.RK23,
        record_norm=False,
        record_max_bdim=True,
        record_average_bdim=False,
        record_total_size=False,
        record_loschmidt_amplitude=False)
    
    tree_bug = BUG(Tree_state, TTNO_Hamil, time_step_size, final_time, operators, config=bug_config)
    tree_bug.run()
    results = tree_bug.results.results
    times = tree_bug.results.times()
    return tree_bug, results, times


def imaginary_time_evolution_TTN(
    tree_state, ttno_imaginary, ttno_physical,
    tau, dtau, max_bond_dim,
):
    if tau <= 0 or dtau <= 0:
        raise ValueError("tau and dtau must be positive")

    n_segments = int(np.ceil(tau / dtau))
    steps_per_segment = 10
    time_step = tau / (n_segments * steps_per_segment)

    state = deepcopy(tree_state)
    state.normalise()

    config = BUGConfig(
        max_bond_dim=max_bond_dim,
        rel_tol=1e-8,
        total_tol=1e-10,
        time_evo_mode=TimeEvoMode.RK23,
        record_max_bdim=False,
    )

    bug = BUG(
        initial_state=state,
        hamiltonian=ttno_imaginary,  # -i H
        time_step_size=time_step,
        final_time=tau,
        operators={},
        config=config,
    )

    energies = []

    for _ in range(n_segments):
        for _ in range(steps_per_segment):
            bug.run_one_time_step()

        # Imaginary-time propagation changes the norm.
        bug.state.normalise()

        energy = float(np.real(
            bug.state.operator_expectation_value(ttno_physical)
        ))
        energies.append(energy)

    ground_state = bug.state
    E_0 = energies[-1]

    print("Physical ground-state energy:", E_0)
    print("Energy by segment:", energies)

    return ground_state, E_0


def krylov_projection_TTN(H,X,vectors,tol=1e-3,Keep_all=False):
    """Takes a physica Hamiltonian in the form of a TTNO and projects it into a low energy subspace without TTNS contraction"""

    m = len(vectors)
    r = X.shape[1]

    assert X.shape[0] == m

    A = ApplicationMethod("direct")

    

    H_vectors = [apply_ttno_to_ttns(vectors[p],H,A)for p in range(m)]

    

    S = np.zeros((m, m),dtype=np.complex128)

    H_eff = np.zeros((m, m),dtype=np.complex128)

    K = np.zeros((m, m),dtype=np.complex128)

    for n in range(m):
        for p in range(m):

            S[n, p] = (vectors[p].scalar_product(vectors[n]))

            H_eff[n, p] = (vectors[p].scalar_product(H_vectors[n]))

            K[n, p] = (H_vectors[p].scalar_product(H_vectors[n]))


    # Transform Hamiltonian into orthonormal
    # Krylov basis

    H_eff_orthog = (X.conj().T@ H_eff@ X)


    # Ritz eigenvalues/eigenvectors


    energies, Q = np.linalg.eigh(H_eff_orthog)
    coefficients = X @ Q

 
    # Residuals
    residuals = np.zeros(r)

    for i in range(r):

        E = energies[i]
        c = coefficients[:, i]

        norm_phi = np.vdot(c,S @ c).real

        H_phi = np.vdot(c,H_eff @ c)

        H2_phi = np.vdot(c,K @ c).real

        residual_squared = (H2_phi- 2.0 * E * H_phi.real+ E**2 * norm_phi)

        residuals[i] = np.sqrt(np.abs(residual_squared))


    # Relative residuals


    relative_residuals = (residuals/ np.maximum(np.abs(energies),1.0))



    if not Keep_all:

        keep = residuals <= tol

        energies = energies[keep]
        Q = Q[:, keep]
        coefficients = coefficients[:, keep]

        residuals = residuals[keep]
        relative_residuals = (
            relative_residuals[keep]
        )
    print("Ritz values and residuals:")
    
    for k, energy in enumerate(energies):
        print(
            f"{k:2d}: "
            f"E = {energy: .10f}, "
            f"residual = {residuals[k]:.3e}, "
            f"relative = {relative_residuals[k]:.3e}"
            )
    return (
        energies,
        Q,
        S,
        coefficients,
        residuals,
        relative_residuals
    )


def pivoted_cholesky(vectors,S, tol=1e-8,svd_tol= 1e-10):
    """"Does a pivoted CHolesky decomposition of the Gram Matrix to ensure linearly independent vectors"""
    n_krylov = len(vectors)

    #S = np.zeros((n_krylov, n_krylov),dtype=np.complex128,)
    #for i in range(n_krylov):
    #    for j in range(n_krylov):
    #        S[i, j] = (vectors[j].scalar_product(vectors[i])) #compute overlap
    n= np.shape(S)[0]
    m=0
    d = np.real(np.diag(S)).copy()
    error= np.sum(d)
    p= np.arange(n)
    l= np.zeros_like(S)
    while error>tol and m<n:
        remaining= p[m:]
        i = m + np.argmax(d[remaining])
        p[m],p[i]=(p[i],p[m])
        pivot=p[m]
        l[m][pivot]= np.sqrt(d[pivot])
        for idx in range(m+1,n):
            term=0
            for jdx in range(m):
                term += np.conjugate(l[jdx][p[m]])*l[jdx][p[idx]]
            l[m][p[idx]]= (S[p[m]][p[idx]]-term)/(l[m][p[m]])
            d[p[idx]] -= np.abs(l[m][p[idx]])**2
        error=sum([d[p[i]] for i in range(m+1,n) ]) 
        m+=1
    L=l[:m]
    U_L, singular_values, Vh = np.linalg.svd(L,full_matrices=False)

    keep = singular_values > svd_tol * singular_values[0]

    sigma = singular_values[keep]

    # V has dimensions (m, r)
    V = Vh.conj().T[:, keep]

    X = V @ np.diag(1.0 / sigma)

    return L, p, d, error,X


def krylov_projection(H,X,vectors,tol=1e-3,Keep_all=False):
    """Takes a Hamiltonian and projects it into an orthonormalised Krylov subspace"""

    # Original nonorthogonal Krylov basis:
    V = np.column_stack([np.asarray(v, dtype=np.complex128)for v in vectors])

    # Projected Hamiltonian in the original nonorthogonal basis
    HV = H @ V
    H_eff = V.conj().T @ HV

    #orthonormalized Krylov basis
    H_orthog = X.conj().T @ H_eff @ X
    energies, Q = np.linalg.eigh(H_orthog) #Diagonalise Effective Hamiltonian

    # Coefficients in the original nonorthogonal Krylov basis
    coefficients = X @ Q

    # Full-space Ritz vectors
    ritz_vectors = V @ coefficients

    residuals = np.empty(len(energies),dtype=float)
    #Determine leakage of Krylov subspace
    relative_residuals = np.empty_like(residuals)

    for k, energy in enumerate(energies):
        phi = ritz_vectors[:, k]

        norm_phi = np.linalg.norm(phi)
        if norm_phi == 0:
            residuals[k] = np.inf
            relative_residuals[k] = np.inf
            continue

        phi = phi / norm_phi
        ritz_vectors[:, k] = phi

        residual_vector = (H @ phi - energy * phi)

        residuals[k] = np.linalg.norm(residual_vector)

        relative_residuals[k] = (residuals[k]/max(1.0, abs(energy)))

    

    if Keep_all:
        keep=None
    else:
        E_scale = 1.0
        
        scaled_residuals = residuals / np.maximum(
                np.abs(energies),
                E_scale,
            )
        keep = scaled_residuals <= tol

        energies=energies[keep]
        Q=Q[:,keep]
        ritz_vectors= ritz_vectors[:,keep]
        residuals= residuals[keep]
        relative_residuals=relative_residuals[keep]

    print("Ritz values and residuals:")

    for k, energy in enumerate(energies):
        print(
            f"{k:2d}: "
            f"E = {energy: .10f}, "
            f"residual = {residuals[k]:.3e}, "
            f"relative = {relative_residuals[k]:.3e}"
        )

    return (np.real_if_close(energies),Q,ritz_vectors,residuals,relative_residuals)

def Complex_time_evolution(
    state, TTNO, time_step_size, final_time,
    max_bond_dim, operators, M,
):
    steps_per_sample = round(final_time / time_step_size)

    if steps_per_sample < 1 or not np.isclose(
        steps_per_sample * time_step_size, final_time
    ):
        raise ValueError(
            "final_time must be an integer multiple of time_step_size"
        )

    config = BUGConfig(
        max_bond_dim=max_bond_dim,
        rel_tol=1e-12,
        total_tol=1e-12,
        time_evo_mode=TimeEvoMode.RK45,
        record_max_bdim=False,
    )

    bug = BUG(
        initial_state=state,
        hamiltonian=TTNO,
        time_step_size=time_step_size,
        final_time=M * final_time,
        operators=operators,
        config=config,
    )

    krylov_states = [deepcopy(bug.state)]

    for step in range(1, M * steps_per_sample + 1):
        bug.run_one_time_step()

        if step % steps_per_sample == 0:
            krylov_states.append(deepcopy(bug.state))

    return krylov_states


 
def Gram_schmidt_TTN(states,tolerance= 1e-10,Keep_all=False):
    """A Gram Schmidt orthonormalisation without TTNS contraction"""
    m = len(states)
    
    S = np.zeros((m, m),dtype=np.complex128)
    if Keep_all:
        tolerance= 0.0
    for i in range(m):
        for j in range(m):
            S[i, j] = states[j].scalar_product(states[i]) #compute overlap

    eigenvalues, U = np.linalg.eigh(S) #diagonalise 

    keep = (eigenvalues> tolerance) #keep positive eigenvalues to ensure full rank

    X = (U[:, keep]@ np.diag(1.0 / np.sqrt(eigenvalues[keep]))) #change to U@ 1/np.sqrt(eigenvalues)

    return X, eigenvalues[keep],S