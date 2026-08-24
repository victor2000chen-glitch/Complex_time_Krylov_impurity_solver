from utils import conjugate
import numpy as np
from pytreenet.ttns.ttns_ttno.application import (
    apply_ttno_to_ttns,
    ApplicationMethod,
)
from scipy import sparse
from pytreenet.time_evolution.bug import BUG, BUGConfig
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.operators import TensorProduct
import copy
from utils import construct_spinful_fermionic_operators


#Time evolution of TTNS
def BUG_time_evolution(Tree_state,TTNO_Hamil,time_step_size,final_time,operators,max_bond_dim=20, rel_tol = 1e-16, abs_tol = 1e-16):
    """Time evolution of TTN using the bug integrator"""
    bug_config = BUGConfig(
        max_bond_dim=max_bond_dim,
        rel_tol=rel_tol,
        total_tol=abs_tol,
        record_norm=True,
        record_max_bdim=True,
        record_average_bdim=True,
        record_total_size=True,
        record_loschmidt_amplitude=True,)
    
    tree_bug = BUG(Tree_state, TTNO_Hamil, time_step_size, final_time, operators, config=bug_config)
    tree_bug.run()
    results = tree_bug.results.results
    times = tree_bug.results.times()
    return tree_bug, results, times



#Gram Schmidt orthonormalisation and Krylov subspace projection
def Grams_schmidt(vectors,tolerance= 1e-10,Keep_all=False):
    """Computes the Gram Matrix and orthonormalises the Hamiltonian"""
    m = len(vectors)

    S = np.zeros(
        (m, m),
        dtype=np.complex128,
    )
    if Keep_all:
        tolerance= 0.0
    else:
        for i in range(m):
            for j in range(m):
                S[i, j] = np.vdot(
                    vectors[i],
                    vectors[j],
                ) #compute overlap

        eigenvalues, U = np.linalg.eigh(S) #diagonalise 

        keep = (eigenvalues> tolerance) #keep positive eigenvalues to ensure full rank

        X = (U[:, keep]@ np.diag(1.0 / np.sqrt(eigenvalues[keep]))) #change to U@ 1/np.sqrt(eigenvalues)

    return X, eigenvalues[keep]


def krylov_projection(H,X,vectors,tol=1e-3,Keep_all=False):
    """Takes an Hamiltonian and projects it into an orthonormalised Krylov subspace"""

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


#def krylov_projection_TTN(H, X, vectors, tol=1e-3, Keep_all=False):

    m = len(vectors)
    r = X.shape[1]
    dict_keys=list(vectors[0].nodes.keys())
    node_key=dict_keys[0]
    assert X.shape[0] == m
    ritz_vectors = copy.deepcopy(vectors)

    A = ApplicationMethod("direct")

    H_eff_orthog = np.zeros((r, r),dtype=np.complex128)

    for i in range(r):
        for j in range(r):

            value = 0.0 + 0.0j

            for n in range(m):
                for p in range(m):

                
                    Hpsi_p = apply_ttno_to_ttns(vectors[p],H,A)

                    
                    H_np = vectors[n].scalar_product(Hpsi_p)

                    value += (np.conjugate(X[n, i])* H_np* X[p, j])

            H_eff_orthog[i, j] = value

    
    energies, Q = np.linalg.eigh(H_eff_orthog)
    coefficients = X @ Q
    print(np.shape(coefficients))

    for i in range(m):
        scalar=0
        for j in range(r):
            scalar += coefficients[i][j]
        operator = TensorProduct({node_key:scalar * np.eye(2)})
        ritz_vectors[i].apply_operator(operator)
    
    #Calculate residuals
    residuals=[]
    phi= copy.deepcopy(ritz_vectors)
    for i in range(r):
        energy= energies[i]
        operator = TensorProduct({node_key:energy * np.eye(2)})
        phi[i].apply_operator(operator)
        residue=phi[i].distance(apply_ttno_to_ttns(ritz_vectors[i],H,A) )
        residuals.append(residue)

    
    return energies, Q,residuals

def krylov_projection_TTN(H,X,vectors,tol=1e-3,Keep_all=False):

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

            S[n, p] = (vectors[n].scalar_product(vectors[p]))

            H_eff[n, p] = (vectors[n].scalar_product(H_vectors[p]))

            K[n, p] = (H_vectors[n].scalar_product(H_vectors[p]))

    # Numerical Hermitisation
    #S = 0.5 * (S + S.conj().T)

    #H_eff = 0.5 * (H_eff + H_eff.conj().T)

    #K = 0.5 * (K + K.conj().T)

    # ------------------------------------------------
    # Transform Hamiltonian into orthonormal
    # Krylov basis
    # ------------------------------------------------

    H_eff_orthog = (X.conj().T@ H_eff@ X)

    #H_eff_orthog = 0.5 * (H_eff_orthog+ H_eff_orthog.conj().T)

    # ------------------------------------------------
    # Ritz eigenvalues/eigenvectors
    # ------------------------------------------------

    energies, Q = np.linalg.eigh(H_eff_orthog)

    # Coefficients in ORIGINAL TTNS basis
    #
    # |Phi_i> = sum_j coefficients[j,i] |psi_j>
    #
    coefficients = X @ Q

    # ------------------------------------------------
    # Residuals
    # ------------------------------------------------

    residuals = np.zeros(r)

    for i in range(r):

        E = energies[i]
        c = coefficients[:, i]

        norm_phi = np.vdot(c,S @ c).real

        H_phi = np.vdot(c,H_eff @ c)

        H2_phi = np.vdot(c,K @ c).real

        residual_squared = (H2_phi- 2.0 * E * H_phi.real+ E**2 * norm_phi)

        residuals[i] = np.sqrt(np.abs(residual_squared))

    # ------------------------------------------------
    # Relative residuals
    # ------------------------------------------------

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
        coefficients,
        residuals,
        relative_residuals
    )

def Complex_time_evolution(TTN,TTNO,time_step_size,final_time,max_bond_dim,operators,M):
    state = TreeTensorNetworkState.from_ttn(TTN)
    Krylov_states=[state]

    for i in range(M):
        print(f"Iteration {i+1} out of {M}")
        tree_bug, results_tilted, times_tilted = BUG_time_evolution(Tree_state=state ,TTNO_Hamil=TTNO,time_step_size=time_step_size,final_time=final_time,
        operators=operators,
        max_bond_dim=max_bond_dim,
        rel_tol=1e-12,
        abs_tol=1e-12)
        Krylov_states.append(tree_bug.state)
        state= tree_bug.state
    return Krylov_states


 
def Gram_schmidt_TTN(states,tolerance= 1e-10,Keep_all=False):
    m = len(states)
    
    S = np.zeros((m, m),dtype=np.complex128)
    if Keep_all:
        tolerance= 0.0
    for i in range(m):
        for j in range(m):
            S[i, j] = states[i].scalar_product(states[j]) #compute overlap

    eigenvalues, U = np.linalg.eigh(S) #diagonalise 

    keep = (eigenvalues> tolerance) #keep positive eigenvalues to ensure full rank

    X = (U[:, keep]@ np.diag(1.0 / np.sqrt(eigenvalues[keep]))) #change to U@ 1/np.sqrt(eigenvalues)

    return X, eigenvalues[keep]



def Greens_function_lehman(c,E,psi,n,w):
    """Constructs the spectral Green's function in the Lehmann representation"""
    c_dag= conjugate(c)
    E_0= E[0]
    psi_0= psi[0]
    total_n= len(E)
    term_1=0
    term_2=0
    G=0.0+0.0j
    for i in range(total_n):
        term_1+= (np.vdot(psi_0,c @ psi[i])*np.vdot(psi[i],c_dag @ psi_0))/(w+1.0j*n+E_0-E[i])
        term_2+= (np.vdot(psi_0,c_dag @ psi[i])*np.vdot(psi[i],c @ psi_0))/(w+1.0j*n+E[i]-E_0)
    G = term_1+term_2
    return G

def spectral_function(G):
    """Calculates the spectralc function from a given Green's function"""
    A= -1/np.pi *np.imag(G)
    return A



def Greens_function_lehman_TTN(E,Ritz_coefficients,Krylo_vectors,eta,w,node_key=None):
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
            C[i][j]= Krylo_vectors[i].scalar_product(C_ttn[j])


    C_dagg= conjugate(C)

    E_0= E[0]
    psi_0= Ritz_coefficients[:,0]
    total_n= len(E)
    term_1=0
    term_2=0
    G=0.0+0.0j
    for i in range(total_n):
        term_1+= (np.vdot(psi_0,C @ Ritz_coefficients[:,i])*np.vdot(Ritz_coefficients[:,i],C_dagg @ psi_0))/(w+1.0j*eta+E_0-E[i])
        term_2+= (np.vdot(psi_0,C_dagg @ Ritz_coefficients[:,i])*np.vdot(Ritz_coefficients[:,i],C @ psi_0))/(w+1.0j*eta+E[i]-E_0)
    G = term_1+term_2
    return G
