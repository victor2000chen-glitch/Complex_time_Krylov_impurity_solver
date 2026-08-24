from fractions import Fraction
from itertools import combinations

import numpy as np

from pytreenet import random
from pytreenet.operators import Hamiltonian, TensorProduct
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.ttno.state_diagram import TTNOFinder
from pytreenet.ttns import TreeTensorNetworkState

import pytreenet.contractions.state_operator_contraction as soc
import pytreenet.contractions.tree_contraction as tree_contraction

def force_ket_first_contraction(ket_node, operator_node):

    return soc.FirstContraction.KET


soc.compare_contr_orders = force_ket_first_contraction




def corrected_final_transpose(
    final_tensor,
    nodes,
    actual_order,
    contraction_order,
):
    indices = {}
    current = 0

    for node_id in contraction_order:
        num_open = nodes[node_id].nopen_legs()

        indices[node_id] = list(
            range(current, current + num_open)
        )

        current += num_open

    permutation = []

    for node_id in actual_order:
        permutation.extend(indices[node_id])

    return final_tensor.transpose(permutation)


tree_contraction.final_transpose = corrected_final_transpose


def conjugate(M):
    Hermit= np.conjugate(M).transpose()
    return Hermit

#Construction of T3NS tree topology
def T3NS(virtual_bond_dimension,bath=False,seed=None):
    """Constructs the T3NS Tree topology"""

    #virtual_bond_dimension=b
    Bath_degrees= 2
    local_physical_dim=2
    ttn = TreeTensorNetworkState()
    root_node,root_tensor = random.random_tensor_node((virtual_bond_dimension ,virtual_bond_dimension ,local_physical_dim) ,"Impurity_1_up",seed=seed)
    ttn.add_root(root_node,root_tensor)

    node1,tensor1 = random.random_tensor_node((virtual_bond_dimension ,virtual_bond_dimension ,virtual_bond_dimension,1),"node1",seed=seed)
    node2,tensor2 = random.random_tensor_node((virtual_bond_dimension ,virtual_bond_dimension ,local_physical_dim),"Impurity_1_down",seed=seed)
    node3,tensor3 = random.random_tensor_node((virtual_bond_dimension,virtual_bond_dimension ,local_physical_dim),"Impurity_2_up",seed=seed)
    node4,tensor4 = random.random_tensor_node((virtual_bond_dimension,virtual_bond_dimension ,local_physical_dim),"Impurity_2_down",seed=seed)
    node5,tensor5 = random.random_tensor_node((virtual_bond_dimension ,virtual_bond_dimension ,local_physical_dim),"Impurity_3_up",seed=seed)
    node6,tensor6 = random.random_tensor_node((virtual_bond_dimension ,virtual_bond_dimension ,local_physical_dim),"Impurity_3_down",seed=seed)
    node7,tensor7 = random.random_tensor_node((virtual_bond_dimension ,virtual_bond_dimension ,virtual_bond_dimension,1),"node2",seed=seed)
    node8,tensor8 = random.random_tensor_node((virtual_bond_dimension ,virtual_bond_dimension ,virtual_bond_dimension,1),"node3",seed=seed)
    node9,tensor9 = random.random_tensor_node((virtual_bond_dimension ,virtual_bond_dimension ,virtual_bond_dimension,1),"node4",seed=seed)
    if bath:
        node10,tensor10 = random.random_tensor_node((virtual_bond_dimension,virtual_bond_dimension ,Bath_degrees),"Bath_1_up_0",seed=seed)
        node11,tensor11 = random.random_tensor_node((virtual_bond_dimension,virtual_bond_dimension ,Bath_degrees),"Bath_1_down_0",seed=seed)
        node12,tensor12 = random.random_tensor_node((virtual_bond_dimension,virtual_bond_dimension ,Bath_degrees),"Bath_2_up_0",seed=seed)
        node13,tensor13 = random.random_tensor_node((virtual_bond_dimension,virtual_bond_dimension ,Bath_degrees),"Bath_2_down_0",seed=seed)
        node14,tensor14 = random.random_tensor_node((virtual_bond_dimension,virtual_bond_dimension ,Bath_degrees),"Bath_3_up_0",seed=seed)
        node15,tensor15 = random.random_tensor_node((virtual_bond_dimension,virtual_bond_dimension ,Bath_degrees),"Bath_3_down_0",seed=seed)
    else: #insert dummy tensors
        node10,tensor10 = random.random_tensor_node((virtual_bond_dimension ,Bath_degrees),"Bath_1_up",seed=seed)
        node11,tensor11 = random.random_tensor_node((virtual_bond_dimension ,Bath_degrees),"Bath_1_down",seed=seed)
        node12,tensor12 = random.random_tensor_node((virtual_bond_dimension ,Bath_degrees),"Bath_2_up",seed=seed)
        node13,tensor13 = random.random_tensor_node((virtual_bond_dimension ,Bath_degrees),"Bath_2_down",seed=seed)
        node14,tensor14 = random.random_tensor_node((virtual_bond_dimension ,Bath_degrees),"Bath_3_up",seed=seed)
        node15,tensor15 = random.random_tensor_node((virtual_bond_dimension ,Bath_degrees),"Bath_3_down",seed=seed)


    ttn.add_child_to_parent (node1,tensor1, 0,"Impurity_1_up",1) #connect Impurity_1_up with node 1
    ttn.add_child_to_parent (node10,tensor10, 0,"Impurity_1_up",1) #connect Impurity_1_up with Bath_up_1
    ttn.add_child_to_parent (node2,tensor2, 0,"node1",1) #connect Impurity_1_down with node 1
    ttn.add_child_to_parent (node5,tensor5, 0,"node1",2) #connect Impurity_3_up with node 1
    ttn.add_child_to_parent (node7,tensor7, 0,"Impurity_1_down",1) #connect Impurity_1_down with node 2
    ttn.add_child_to_parent (node3,tensor3, 0,"node2",1) #connect Impurity_2_up with node 2
    ttn.add_child_to_parent (node11,tensor11, 0,"node2",2) #connect Bath_1_down with node 2
    ttn.add_child_to_parent (node8,tensor8, 0,"Impurity_2_up",1) #connect Impurity_2_up with node 3
    ttn.add_child_to_parent (node4,tensor4, 0,"node3",1) #connect Impurity_2_down with node 3
    ttn.add_child_to_parent (node12,tensor12, 0,"node3",2) #connect Bath_2_up with node 3
    ttn.add_child_to_parent (node13,tensor13, 0,"Impurity_2_down",1) #connect Bath_2_down with Impurity 2 down
    ttn.add_child_to_parent (node9,tensor9, 0,"Impurity_3_up",1) #connect Impurity_3_up with node 4
    ttn.add_child_to_parent (node6,tensor6, 0,"node4",1) #connect Impurity_3_down with node 4
    ttn.add_child_to_parent (node14,tensor14, 0,"node4",2) #connect Bath_3_up with node 4
    ttn.add_child_to_parent (node15,tensor15, 0,"Impurity_3_down",1) #connect Bath_3_down with Impurity_3_down
    return ttn

#Contstruct impurity Hamiltonian as TTNO
def construct_Hubbard_Kanamori_hamiltonian(
    norb,
    U,
    U_prime,
    J,
    angle_factor=0.0,
):
    """Constructs the Hubbard Kanamori Impurity Hamiltonian for TTNO"""
    
    contour_factor= np.exp(angle_factor*1.0j)

    annihilate = np.array(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=complex,
    )
    pz= np.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=complex,
    )

    create = annihilate.conj().T
    number = create @ annihilate

    conversion_dictionary = {
        "c": annihilate,
        "c_dag": create,
        "n": number,
        "I1": np.eye(1, dtype=complex),
        "I2": np.eye(2, dtype=complex),
        "pz": pz,
    }

    coefficient_dict = {
        "U": contour_factor*U,
        "U_prime": contour_factor*U_prime,
        "U_prime_minus_J": contour_factor*(U_prime - J),
        "J": contour_factor*J,
        "contour_factor":contour_factor,
    }

    hamiltonian = Hamiltonian(
        terms=[],
        conversion_dictionary=conversion_dictionary,
        coeffs_mapping=coefficient_dict,
    )

    #Hubbard term
    for i in range(1, norb + 1): 
        term = TensorProduct({
            f"Impurity_{i}_up": "n",
            f"Impurity_{i}_down": "n",
        })

        hamiltonian.add_term((
            Fraction(1),
            "U",
            term,
        ))

    for i, j in combinations(range(1, norb + 1), 2):
        i_up = f"Impurity_{i}_up"
        i_down = f"Impurity_{i}_down"
        j_up = f"Impurity_{j}_up"
        j_down = f"Impurity_{j}_down"
        

        # Opposite-spin terms:
        
        hamiltonian.add_term((
            Fraction(1),
            "U_prime",
            TensorProduct({
                i_up: "n",
                j_down: "n",
            }),
        ))

        hamiltonian.add_term((
            Fraction(1),
            "U_prime",
            TensorProduct({
                i_down: "n",
                j_up: "n",
            }),
        ))

        # Same-spin density terms:
        
        hamiltonian.add_term((
            Fraction(1),
            "U_prime_minus_J",
            TensorProduct({
                i_up: "n",
                j_up: "n",
            }),
        ))

        hamiltonian.add_term((
            Fraction(1),
            "U_prime_minus_J",
            TensorProduct({
                i_down: "n",
                j_down: "n",
            }),
        ))

        # Spin-flip term
        hamiltonian.add_term((
            Fraction(-1),
            "J",
            TensorProduct({
                i_up: "c_dag",
                i_down: "c",
                j_up: "c",
                j_down: "c_dag",
            }),
        ))

        # Hermitian conjugate
        hamiltonian.add_term((
            Fraction(-1),
            "J",
            TensorProduct({
                i_up: "c",
                i_down: "c_dag",
                j_up: "c_dag",
                j_down: "c",
            }),
        ))

        # Pair hopping
        hamiltonian.add_term((
            Fraction(1),
            "J",
            TensorProduct({
                i_up: "c_dag",
                i_down: "c_dag",
                j_up: "c",
                j_down: "c",
            }),
        ))

        # Hermitian conjugate
        hamiltonian.add_term((
            Fraction(1),
            "J",
            TensorProduct({
                i_up: "c",
                i_down: "c",
                j_up: "c_dag",
                j_down: "c_dag",
            }),
        ))

    return hamiltonian


def add_bath_chain(
    ttn,
    parent_id,
    number_bath_sites,
    bond_dim=4,
):

    """Adds Bath in the form of MPS chains to the TTNS"""
    # turns "Bath_1_up_0" into "Bath_1_up"
    prefix = parent_id.rsplit("_", 1)[0]

    current_parent = parent_id

    # Include the already existing site 0.
    bath_ids = [parent_id]

    # Add sites 1, ..., number_bath_sites - 1.
    for bath_index in range(1, number_bath_sites):
        node_id = f"{prefix}_{bath_index}"
        print(node_id)

        is_last = (
            bath_index == number_bath_sites - 1
        )

        # Leg 1 of the current node is its outgoing virtual leg.
        incoming_dim = (
            ttn.tensors[current_parent].shape[1]
        )

        if is_last: #Check to see if its last site or not
            # Incoming virtual leg + physical leg
            shape = (
                incoming_dim,
                2,
            )
        else:
            # Incoming virtual leg + outgoing virtual leg
            # + physical leg
            shape = (
                incoming_dim,
                bond_dim,
                2,
            )

        bath_node, bath_tensor = (
            random.random_tensor_node(
                shape,
                node_id,
            )
        )

        ttn.add_child_to_parent(
            bath_node,
            bath_tensor,
            child_leg=0,
            parent_id=current_parent,
            parent_leg=1,
        )

        bath_ids.append(node_id)
        current_parent = node_id

    return bath_ids




def construct_fermionic_order(norb,n_bath):
    """
    Global JW ordering.

    Impurities come first so that the usual local
    Hubbard-Kanamori representation remains string-free.

    Bath modes are ordered as spinful bath levels:
    Bath_m_up_k, Bath_m_down_k
    """

    order = []

    # Impurity modes
    for orbital in range(1, norb + 1):
        order.extend([
            f"Impurity_{orbital}_up",
            f"Impurity_{orbital}_down",
        ])

    # Bath modes
    for orbital in range(1, norb + 1):
        for bath_index in range(n_bath):
            order.extend([
                f"Bath_{orbital}_up_{bath_index}",
                f"Bath_{orbital}_down_{bath_index}",
            ])
    node_order=["node1","node2","node3","node4"]
    contraction_order= order+ node_order #define contraction order of Tensor network
    return order,contraction_order



def jw_bilinear(fermionic_order,creation_site,annihilation_site,):
    """
    JW representation of

        f_creation_site^dagger f_annihilation_site.
    """
    #Creation site and annihilation site are strings denoting on which site these operators are acting 
    positions = { 
        site: index
        for index, site in enumerate(fermionic_order)
    } #Dictionary where each index is identified with a site

    if creation_site not in positions:
        raise KeyError(
            f"{creation_site!r} is absent from fermionic_order."
        )

    if annihilation_site not in positions:
        raise KeyError(
            f"{annihilation_site!r} is absent from fermionic_order."
        )

    p = positions[creation_site]
    q = positions[annihilation_site]

    if p == q: #Same site
        return TensorProduct({
            creation_site: "n",
        })

    local_operators = {creation_site: "c_dag",annihilation_site: "c",}

    for index in range(
        min(p, q) + 1,
        max(p, q),
    ):
        intermediate_site = fermionic_order[index] #sites inbetween 
        local_operators[intermediate_site] = "pz" #add pauli z 

    return TensorProduct(local_operators) #returns a tensorproduct of JW strings





def add_bath_interaction(Hamiltonian,n_bath,v,e):
    """Add Interactions with a fermionic bath to the Impurity Hamiltonian"""
    coeff_dict=Hamiltonian.coeffs_mapping
    contour_factor= coeff_dict["contour_factor"]
    norb=3
    fermionic_order,_ = construct_fermionic_order(norb=norb,n_bath=n_bath) #JW ordering

    #for i in range(n_bath):
    #    coeff_dict[f"v_{i}"]=contour_factor*v[i]
    #    coeff_dict[f"e_{i}"]=contour_factor*e[i]

    for orbital in range(1, norb + 1): #Orbital index
        for spin in ("up", "down"): #Spin index
            impurity_id = (f"Impurity_{orbital}_{spin}")

            for bath_index in range(n_bath): #define keys for coefficients
                bath_id = (
                    f"Bath_{orbital}_{spin}_{bath_index}"
                )

                energy_key = (
                    f"e_{orbital}_{spin}_{bath_index}"
                )

                forward_key = (
                    f"v_forward_{orbital}_{spin}_{bath_index}"
                )

                backward_key = (
                    f"v_backward_{orbital}_{spin}_{bath_index}"
                )

                # q e_k
                coeff_dict[energy_key] = (contour_factor* e[bath_index]) #Add bath hybridisation coefficients and on site energies to dictionary

                # q v_k
                coeff_dict[forward_key] = (contour_factor* v[bath_index])

                # q v_k*
                #
                # Do not conjugate contour_factor. The complete
                # physical Hamiltonian is multiplied by q.
                coeff_dict[backward_key] = (contour_factor* np.conjugate(v[bath_index])) #currently v is purley real 

                # On site one body hubbard terms
                Hamiltonian.add_term((
                    Fraction(1),
                    energy_key,
                    TensorProduct({
                        bath_id: "n",
                    }),
                )) #No need for JW strings for n operator
                #Bath hopping terms

                # v_k d† c_k
                Hamiltonian.add_term((
                    Fraction(1),
                    forward_key,
                    jw_bilinear( #include JW transformation
                        fermionic_order=fermionic_order,
                        creation_site=impurity_id,
                        annihilation_site=bath_id,
                    ),
                ))

                # v_k* c†_k d
                Hamiltonian.add_term((
                    Fraction(1),
                    backward_key,
                    jw_bilinear(
                        fermionic_order=fermionic_order,
                        creation_site=bath_id,
                        annihilation_site=impurity_id,
                    ),
                ))



    return


def SIAM_Hamiltonian(state,angle,eshift,n_bath,hop,eb,ei,U):
    hamiltonian = Hamiltonian()
    c = np.array([[0, 1], [0, 0]])
    n = np.array([[0, 0], [0, 1]])
    pz = np.array([[1, 0], [0, -1]])
    conversion_dict = {}
    conversion_dict["I2"] = np.eye(2)
    conversion_dict["I1"] = np.eye(1)
    conversion_dict["c"] = c
    conversion_dict["c*"] = c.T
    conversion_dict["n"] = n
    conversion_dict["pz"] = pz
    frac = Fraction(1)
    coeff_map = dict()
    angle_fact = np.cos(angle) + 1j * np.sin(angle)
    angle_fact = np.real_if_close(angle_fact)
    angle_fact = -1j*(np.real_if_close(1j*(angle_fact)))
    e_shift_term = TensorProduct()
    e_shift_coeff_id = "E"
    coeff_map[e_shift_coeff_id] = eshift * angle_fact
    hamiltonian.add_term((frac, e_shift_coeff_id, e_shift_term))
    
    for i in range(n_bath):

        pzs_up = {f"bath{j}(up)": "pz" for j in range(i + 1, n_bath)}

        # Hopping term from impurity-site to bath-site i for up-spin.
        hop_term_up = TensorProduct(
            {f"bath{i}(up)": "c*"} | pzs_up | {"imp(up)": "c"}
        )
        hop_coeff_id = f"t{i}*"
        coeff_map[hop_coeff_id] = hop[i] * angle_fact
        hamiltonian.add_term((frac, hop_coeff_id, hop_term_up))

        # Hopping term bath-site i to impurity-site for up-spin.
        hop_term_up_dag = TensorProduct(
            {f"bath{i}(up)": "c"} | pzs_up | {"imp(up)": "c*"}
        )
        hop_conj_coeff_id = f"t{i}"
        coeff_map[hop_conj_coeff_id] = np.conj(hop[i]) * angle_fact
        hamiltonian.add_term((frac, hop_conj_coeff_id, hop_term_up_dag))

        # On-site energy term for bath-site i.
        os_term_up = TensorProduct({f"bath{i}(up)": "n"})
        os_coeff_id = f"e{i}"
        coeff_map[os_coeff_id] = eb[i] * angle_fact
        hamiltonian.add_term((frac, os_coeff_id, os_term_up))

        ######
        # Same terms, but for spin down. 
        # Jordan-Wigner order differs though.
        pzs_down = {f"bath{j}(down)": "pz" for j in range(0, i)}


        hop_term_down = TensorProduct(
            {"imp(down)": "c"} | pzs_down | {f"bath{i}(down)": "c*"}
        )
        hamiltonian.add_term((frac, hop_coeff_id, hop_term_down))


        hop_term_down_dag = TensorProduct(
            {"imp(down)": "c*"} | pzs_down | {f"bath{i}(down)": "c"}
        )
        hamiltonian.add_term((frac, hop_conj_coeff_id, hop_term_down_dag))


        os_term_down = TensorProduct({f"bath{i}(down)": "n"})
        hamiltonian.add_term((frac, os_coeff_id, os_term_down))

    # both terms have the same coefficient (chemical potential)
    os_imp_coeff_id = "eimp"
    coeff_map[os_imp_coeff_id] = ei * angle_fact

    # up-spin
    os_term_up = TensorProduct({"imp(up)": "n"})
    hamiltonian.add_term((frac, os_imp_coeff_id, os_term_up))

    # down-spin
    os_term_down = TensorProduct({"imp(down)": "n"})
    hamiltonian.add_term((frac, os_imp_coeff_id, os_term_down))
    inter_term = TensorProduct({"imp(up)": "n", "imp(down)": "n"})
    inter_coeff_id = "U"
    coeff_map["U"] = U * angle_fact
    hamiltonian.add_term((frac, inter_coeff_id, inter_term))
    hamiltonian.conversion_dictionary = conversion_dict
    hamiltonian.coeffs_mapping = coeff_map

    siam_ttno = TreeTensorNetworkOperator.from_hamiltonian(
        hamiltonian=hamiltonian, reference_tree=state, method=TTNOFinder.SGE
    )



    return siam_ttno
