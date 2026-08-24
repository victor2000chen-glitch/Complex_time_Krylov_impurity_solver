
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


def conjugate(M):
    Hermit= np.conjugate(M).transpose()
    return Hermit

def construct_spinful_fermionic_operators(nsites: int):
    """Constructs Fermionic operators from qubit operators using JW transformation"""

    nmodes = 2 * nsites

    id2 = sparse.identity(2, format="csr")
    z   = sparse.csr_matrix([[1.,0.],[0.,-1.]])
    u   = sparse.csr_matrix([[0.,0.],[1.,0.]])

    create = []

    for i in range(nmodes):

        c = sparse.identity(1, format="csr")

        for j in range(nmodes):

            if j < i:
                c = sparse.kron(c, z, format="csr") #jordan Wigner transformation

            elif j == i:
                c = sparse.kron(c, u, format="csr")

            else:
                c = sparse.kron(c, id2, format="csr")

        c.eliminate_zeros()
        create.append(c)

    annihilate = [c.conj().T.tocsr() for c in create]


    number = []

    for i in range(nmodes):

        f = 1 << (nmodes - i - 1)

        data = [
            1.0 if (n & f) else 0.0
            for n in range(2**nmodes)
        ]

        number.append(
            sparse.dia_matrix((data, 0),
                              shape=(2**nmodes, 2**nmodes)).tocsr()
        )

    clist = []
    alist = []
    nlist = []

    for site in range(nsites):

        up = 2 * site
        dn = up + 1

        clist.append((create[up], create[dn]))
        alist.append((annihilate[up], annihilate[dn]))
        nlist.append((number[up], number[dn]))

    return clist, alist, nlist


def Hubbard_Kanamori_interaction_sparse(norb, U, U_prime, J):
    """Constructs the HK Hamiltonian as a sparse matrix"""



    nmodes = 2 * (norb)
    dim = 2 ** nmodes

    c, a, n = construct_spinful_fermionic_operators(norb)

    H = sparse.csr_matrix((dim, dim), dtype=np.complex128)

    
    #Intraorbital Hubbard interaction
    for m in range(norb):
        H += U * (n[m][0] @ n[m][1])


    #Interorbital interactions

    for m in range(norb):
        for mp in range(m + 1, norb):

            #opposite spin density interaction
            H += U_prime * (n[m][0] @ n[mp][1]+ n[m][1] @ n[mp][0])

            
            #same spin density interaction
            H += (U_prime - J) * (n[m][0] @ n[mp][0]+ n[m][1] @ n[mp][1])

            #Hund spin flip
            H += -J * (c[m][0] @ a[m][1] @ c[mp][1] @ a[mp][0]+c[m][1] @ a[m][0] @ c[mp][0] @ a[mp][1])

            # Pair hopping
            H += J * (c[m][0] @ c[m][1] @ a[mp][1] @ a[mp][0]+c[mp][0] @ c[mp][1] @ a[m][1] @ a[m][0])


    return H


def Hubbard_Kanamori_interaction_with_bath_sparse(
    norb,
    n_bath,
    U,
    U_prime,
    J,
    v,
    e,
):
    """
    Sparse Hubbard-Kanamori Hamiltonian with n_bath spinful
    bath levels per impurity orbital.

    Ordering of spinful sites:
        Impurity_1, ..., Impurity_norb,
        Bath_1_0, ..., Bath_1_(n_bath-1),
        Bath_2_0, ..., Bath_2_(n_bath-1),

    """


    # Convert scalar parameters to arrays when necessary.
    v = np.asarray(v, dtype=np.complex128)
    e = np.asarray(e, dtype=np.complex128)

    if v.ndim == 0:
        v = np.full(
            n_bath,
            v.item(),
            dtype=np.complex128,
        )
    else:
        v = v.reshape(-1)

    if e.ndim == 0:
        e = np.full(
            n_bath,
            e.item(),
            dtype=np.complex128,
        )
    else:
        e = e.reshape(-1)


    if len(v) != n_bath:
        raise ValueError(
            f"v must have length {n_bath}; received {len(v)}."
        )

    if len(e) != n_bath:
        raise ValueError(
            f"e must have length {n_bath}; received {len(e)}."
        )

    # There are n_bath spinful bath sites for every orbital.
    total_bath_sites = norb * n_bath
    number_spinful_sites = norb + total_bath_sites

    nmodes = 2 * number_spinful_sites
    dim = 2**nmodes

    # c = creation, a = annihilation, n = number.
    # All operators already contain global JW strings.
    c, a, n = construct_spinful_fermionic_operators(
        number_spinful_sites
    )

    H = sparse.csr_matrix(
        (dim, dim),
        dtype=np.complex128,
    )

  
    # Intra-orbital Hubbard interaction

    for m in range(norb):
        H += U * (
            n[m][0] @ n[m][1]
        )

  
    # Inter-orbital Kanamori interaction

    for m in range(norb):
        for mp in range(m + 1, norb):

            # Opposite-spin density interaction
            H += U_prime * (
                n[m][0] @ n[mp][1]
                + n[m][1] @ n[mp][0]
            )

            # Same-spin density interaction
            H += (U_prime - J) * (
                n[m][0] @ n[mp][0]
                + n[m][1] @ n[mp][1]
            )

            # Hund spin flip
            H += -J * (
                c[m][0]
                @ a[m][1]
                @ c[mp][1]
                @ a[mp][0]

                +

                c[m][1]
                @ a[m][0]
                @ c[mp][0]
                @ a[mp][1]
            )

            # Pair hopping
            H += J * (
                c[m][0]
                @ c[m][1]
                @ a[mp][1]
                @ a[mp][0]

                +

                c[mp][0]
                @ c[mp][1]
                @ a[m][1]
                @ a[m][0]
            )


    # Bath onsite energies and hybridization

    for orbital in range(norb):
        for bath_index in range(n_bath):

            # Spinful-site index of this bath level.
            bath_site = (
                norb
                + orbital * n_bath
                + bath_index
            )

            for spin in (0, 1):

                H += (
                    e[bath_index]
                    * n[bath_site][spin]
                )

                H += (
                    v[bath_index]
                    * c[orbital][spin]
                    @ a[bath_site][spin]
                )

                H += (
                    np.conjugate(v[bath_index])
                    * c[bath_site][spin]
                    @ a[orbital][spin]
                )

    H.sum_duplicates()
    H.eliminate_zeros()

    return H

#Exact imaginary and real time evolution
def exact_time_evolution(state, Hamil, T):
    state = np.asarray(state, dtype=complex)
    state = state / np.linalg.norm(state)

    psi_T = expm_multiply(
        -1.0j*T* Hamil,
        state,
    )
    norm = np.linalg.norm(psi_T)
    psi_T /= norm
    energy = np.vdot(psi_T,Hamil @ psi_T,).real

    return energy, psi_T

def exact_imaginary_time_evolution(state,H,tau,normalize_output=True):

    psi_0 = np.asarray(state, dtype=np.complex128)
    psi_0 /= np.linalg.norm(psi_0)

    psi_tau = expm_multiply(
        -tau * H,
        psi_0,
    )

    norm_squared = np.vdot(
        psi_tau,
        psi_tau,
    ).real

    norm = np.sqrt(norm_squared)


    energy = np.vdot(
        psi_tau,
        H @ psi_tau,
    ) / norm_squared

    energy = np.real_if_close(energy)

    if normalize_output:
        psi_tau = psi_tau / norm

    return energy, psi_tau, norm

def state_to_vector(state, node_order):
    """Contracts a tree tensor network and returns a vector"""
    tensor, returned_order = state.completely_contract_tree(
        to_copy=True,
        order=node_order,
    )

    return np.asarray(tensor, dtype=complex).reshape(-1)


def Hybridisation_function(D, w):
    """"Semicircular Hybridisation function"""
    delta= D/(2* np.pi) * np.sqrt(1-(w/D)**2)
    return delta 


def Discretisation_function(D, n_bath):
    """Discretisation of Bath"""
    v=[]
    grid=np.linspace(-D, D, num=n_bath+2,endpoint=True)
    width= (2*D)/(n_bath+1)
    bath_energies = grid[1:-1]
    for w in grid:
        delta= Hybridisation_function(D, w)
        #bath_energies.append(delta)
        area= delta* width
        v.append(np.sqrt(area))
    #remove zeros
    v_d= v[1:-1]
    #bath_energies.pop(0)
    #bath_energies.pop(-1)
    return v_d,bath_energies