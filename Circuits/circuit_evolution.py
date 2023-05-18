
import numpy as np
L_max = 60
T_max = 300


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


