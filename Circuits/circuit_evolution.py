import numpy as np

def evenlayer(state,U_list,L):
    L_A = len(state.shape) - L
    state = state.reshape((4,)*(L//2) + (2,)*L_A)
    for i in range(0,L-1,2):          
        U = U_list[i//2]
        state = np.tensordot(U,state,axes=(-1,i//2))
        state = np.moveaxis(state,0,i//2)
    state = state.reshape((2,)*(L+L_A))
    return state


def oddlayer(state,U_list,L,BC='PBC'):
    L_A = len(state.shape) - L
    state = np.moveaxis(state,L-1,0)
    state = state.reshape((4,)*(L//2) + (2,)*L_A)
    for i in range(1,L-1,2):          
        U = U_list[(i+1)//2]
        state = np.tensordot(U,state,axes=(-1,(i+1)//2))
        state = np.moveaxis(state,0,(i+1)//2)
    if BC == 'PBC' and L%2 == 0:
        U = U_list[0]
        state = np.tensordot(U,state,axes=(-1,0))
    state = state.reshape((2,)*(L+L_A))
    state = np.moveaxis(state,0,L-1)
    return state


def non_local_gate(state: np.ndarray,M: np.ndarray, location: list):
    """Perform non_local gate which may or maynot be unitary

    Args:
        state (np.ndarray): (2,)*L shaped array where L is system size
        M: (Non-)unitary gate to be applied
        location (list): sites where the gate acts
    
    output:
        new_state: (2,)*L shaped array with the Operator M acted on
    """
    measurement_dim = M.shape[0]
    num_of_sites = len(location)
    L = len(state.shape)

    assert location == sorted(location), print('Locations provided must be sorted in ascending order')
    assert L>1, print("Shape of state has only one axis. state should have (2,)*L where L is the system size")
    assert int(2**num_of_sites) == int(measurement_dim), print('Number of location points don\'t match with Kraus operators dimension')

    for i,x in enumerate(location):
        state = np.swapaxes(state,x,i)

    state = np.reshape(state,(2**num_of_sites,)+(2,)*(L-num_of_sites))


    new_state = np.tensordot(M,state,axes=(-1,0))
    norm = np.sum(np.abs(new_state)**2)

    # Keeping the state same as in the input!
    new_state = np.reshape(new_state,(2,)*L)/norm**0.5
    state = np.reshape(state,(2,)*L)
    for i,x in enumerate(location):
        state = np.swapaxes(state,x,i)
        new_state = np.swapaxes(new_state,x,i)
    
    return new_state


def generalized_measurement(state: np.ndarray,kraus_operators: list, location: list, rng: np.random.default_rng):
    """Perform genreal POVM

    Args:
        state (np.ndarray): (2,)*L shaped array where L is system size
        kraus_operators (list): list of Kraus operators for measurement
        location (list): where the POVM acts
        rng: random number generator
    
    output:
        new_state: (2,)*L shaped array with a Kraus Operator acted on (the operator is determined by Born rule)
        outcome: index of the Kraus operator applied.
    """
    measurement_dim = kraus_operators[0].shape[0]
    num_of_sites = len(location)
    L = len(state.shape)

    assert location == sorted(location), print('Locations provided must be sorted in ascending order')

    assert L>1, print("Shape of state has only one axis. state should have (2,)*L where L is the system size")
    assert np.all([i.shape == (measurement_dim,measurement_dim) for i in kraus_operators]), print("Kraus operators are of not same dimension")
    assert int(2**num_of_sites) == int(measurement_dim), print('Number of location points don\'t match with Kraus operators dimension')

    for i,x in enumerate(location):
        state = np.swapaxes(state,x,i)

    state = np.reshape(state,(2**num_of_sites,)+(2,)*(L-num_of_sites))

    # normalize the kraus operators
    norm = np.zeros((measurement_dim,measurement_dim))
    for K in kraus_operators:
        norm += np.dot(np.transpose(np.conj(K)),K)
    assert norm/norm[0,0] == np.identity(measurement_dim), print('POVM not properly defined. The POVMs normalize to %'.format(norm))

    random_number = rng.uniform(0,1)
    cum_probs = 0
    for i in range(len(kraus_operators)):
        new_state = np.tensordot(kraus_operators[i]/norm[0,0]**0.5,state,axes=(-1,0))
        prob = np.sum(np.abs(new_state)**2)
        cum_probs += prob
        if cum_probs > random_number:
            break
    outcome = i
    # Keeping the state same as in the input!
    new_state = np.reshape(new_state,(2,)*L)/prob**0.5
    state = np.reshape(state,(2,)*L)
    for i,x in enumerate(location):
        state = np.swapaxes(state,x,i)
        new_state = np.swapaxes(new_state,x,i)
    
    return new_state, outcome

def measurement_layer(state,m_locations,rng_outcome: np.random.default_rng):
    for m in m_locations:
        state = np.swapaxes(state,m,0)
        p_0 = np.sum(np.abs(state[0,:])**2)
        
        if rng_outcome.uniform(0,1) < p_0:
            outcome = 0
        else:
            outcome = 1
        
        if outcome == 0:
            state[1,:]=0
        elif outcome == 1:
            state[0,:] = 0
        S = np.sum(np.abs(state.flatten())**2)
        state = state/S**0.5
        state = np.swapaxes(state,0,m)
    
    return state



def weak_measurement_layer(state,theta,L:int,rng_outcome: np.random.default_rng, m_locations=None):
    """To implement exp{-i*theta/2 [1-Z_q]X_qa} = exp{-i*theta X_qa/2} exp{i*theta/2 Z_q*X_qa} on physical qubits. This performs weak measurement.

    Args:
        theta (_float): measurement strength. theta = 0: no measurement. theta=pi/2: projective measurement
        L (_int): system size
        m_locations (list): measurement_locations. Default: measure all locations

    Returns:
        _type_: _description_
    """
    if m_locations is None:
        m_locations = list(range(L))
    
    ## Implementing weak measurement
    for m in m_locations:
        state = np.swapaxes(state,m,0)
        p_0 = np.sum(np.abs(state[0,:])**2)
        p_1 = 1-p_0
        p_0a = p_0 + p_1 * np.cos(theta) 
        if rng_outcome.uniform(0,1) < p_0a:
            outcome = 0
        else:
            outcome = 1
        
        if outcome == 0:
            state[1,:] = state[1,:]*np.cos(theta)
        elif outcome == 1:
            state[0,:] = 0
        S = np.sum(np.abs(state.flatten())**2)
        state = state/S**0.5
        state = np.swapaxes(state,0,m)
    return state


