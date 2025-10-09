import numpy as np

def get_nan_gradient(F, dt, axis = -1):
    '''
    This function calculates the gradient of a numpy array, while ignoring NaN values.

    Parameters
    ----------
    F : np.ndarray
        The input array. Shape is [..., M].
    dt : float
        The time step size.
    axis : int, optional
        The axis along which to calculate the gradient. The default is -1.

    Returns
    -------
    np.ndarray
        The gradient of the input array.

    '''
    # Move axis to the last dimension
    F = np.moveaxis(F, axis, -1)

    # Find contiguous segments of non-NaN values
    missing_timesteps_start = np.isfinite(F).argmax(-1) # [...]
    missing_timesteps_end = missing_timesteps_start + np.isfinite(F).sum(-1) # [...]

    missing_timesteps = np.stack([missing_timesteps_start, missing_timesteps_end], -1) # [..., 2]
    missing_timesteps = np.unique(missing_timesteps.reshape(-1,2), axis = 0) # n, 2 
    
    dF = np.zeros_like(F) # [..., M]
    for (missing_timestep_start, missing_timestep_end) in missing_timesteps:
        mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end) # [...]
        
        if np.abs(missing_timestep_end - missing_timestep_start) > 1:
            f_mask = F[mask] # [N, M]
            f_mask_time = f_mask[:,missing_timestep_start:missing_timestep_end] # [N,m]
            assert np.isfinite(f_mask_time).all(), "There are still NaN values in the original data."
            df_mask_time = np.gradient(f_mask_time, axis = -1) # [N,m]
            assert np.isfinite(df_mask_time).all(), "There are still NaN values in the derivatives."
            df_mask = np.zeros_like(f_mask) # [N, M]
            df_mask[:,missing_timestep_start:missing_timestep_end] = df_mask_time
            dF[mask] = df_mask
            
    dF[~np.isfinite(F)] = np.nan

    # Revert the axis move
    dF = np.moveaxis(dF, -1, axis)
    return dF / dt

def get_nan_integral(dF, f0, dt, axis = -1):
    '''
    This function calculates the integral of a numpy array, while ignoring NaN values.

    Parameters
    ----------
    F : np.ndarray
        The input array. Shape is [..., M].
    f0 : np.ndarray
        The initial value for the integral. Shape is [..., M].
    dt : float
        The time step size.
    axis : int, optional
        The axis along which to calculate the integral. The default is -1.

    Returns
    -------
    np.ndarray
        The integral of the input array.

    '''
    # Move axis to the last dimension
    dF = np.moveaxis(dF, axis, -1)
    f0 = np.moveaxis(f0, axis, -1)

    # Find contiguous segments of non-NaN values
    missing_timesteps_start = np.isfinite(dF).argmax(-1) # [...]
    missing_timesteps_end = missing_timesteps_start + np.isfinite(dF).sum(-1) # [...]

    missing_timesteps = np.stack([missing_timesteps_start, missing_timesteps_end], -1) # [..., 2]
    missing_timesteps = np.unique(missing_timesteps.reshape(-1,2), axis = 0) # n, 2 
    
    intF = np.zeros_like(dF) # [..., M]
    for (missing_timestep_start, missing_timestep_end) in missing_timesteps:
        mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end) # [...]
        
        if np.abs(missing_timestep_end - missing_timestep_start) > 1:
            f_mask = dF[mask] # [N, M]
            f0_mask = f0[mask] # [N, M]
            f_mask_time = f_mask[:,missing_timestep_start:missing_timestep_end] # [N,m]
            f0_mask_time = f0_mask[:,missing_timestep_start] # [N,]
            assert np.isfinite(f_mask_time).all(), "There are still NaN values in the original data."
            assert np.isfinite(f0_mask_time).all(), "There are still NaN values in the initial values."

            # get mean between timesteps
            f_mask_time = 0.5 * (f_mask_time[:, :-1] + f_mask_time[:, 1:]) # [N, m-1]
            cum_sum = np.cumsum(f_mask_time, axis = -1) # [N,m-1]
            # Prepend zero
            cum_sum = np.concatenate([np.zeros_like(f0_mask_time[..., np.newaxis]), cum_sum], axis = -1) # [N,m]
            # calculate integral
            intf_mask_time = cum_sum * dt + f0_mask_time[..., np.newaxis] # [N,m]
            assert np.isfinite(intf_mask_time).all(), "There are still NaN values in the integrals."
            intf_mask = np.zeros_like(f_mask) # [N, M]
            intf_mask[:,missing_timestep_start:missing_timestep_end] = intf_mask_time
            intF[mask] = intf_mask
            
    intF[~np.isfinite(dF)] = np.nan

    # Revert the axis move
    intF = np.moveaxis(intF, -1, axis)
    return intF


def get_derivatives(P, dt, data_type):
    assert P.shape[-1] == len(data_type), "The last dimension of P should correspond to the length of data_type"
    assert data_type[0] == 'x' and data_type[1] == 'y', "The first two elements of data_type should be 'x' and 'y'"
    # Collapse sample/agent dimensions
    p_shape = P.shape
    P = P.reshape(-1, P.shape[-2], len(data_type))

    # Do marginal velocities first
    if 'v_x' in data_type:
        i_vx = data_type.index('v_x')

        P_v_x = get_nan_gradient(P[..., 0], dt, axis = -1)
        P[..., i_vx] = P_v_x

    if 'v_y' in data_type:
        i_vy = data_type.index('v_y')
        P_v_y = get_nan_gradient(P[..., 1], dt, axis = -1)
        P[..., i_vy] = P_v_y
    
    # Do marginal accelerations, based on previous velocities
    if 'a_x' in data_type:
        i_ax = data_type.index('a_x')
        # If velocities are required, they will have been calculated already
        if 'v_x' in data_type:
            i_vx = data_type.index('v_x')
            P_vx = P[..., i_vx]
        else:
            P_vx = get_nan_gradient(P[..., 0], dt, axis = -1)
        P_ax = get_nan_gradient(P_vx, dt, axis = -1)
        P[..., i_ax] = P_ax
    
    if 'a_y' in data_type:
        i_ay = data_type.index('a_y')
        # If velocities are required, they will have been calculated already
        if 'v_y' in data_type:
            i_vy = data_type.index('v_y')
            P_vy = P[..., i_vy]
        else:
            P_vy = get_nan_gradient(P[..., 1], dt, axis = -1)
        P_ay = get_nan_gradient(P_vy, dt, axis = -1)
        P[..., i_ay] = P_ay

    # Do total velocities, based on previous velocities
    if 'v' in data_type:
        i_v = data_type.index('v')
        if 'v_x' in data_type:
            i_vx = data_type.index('v_x')
            P_vx = P[..., i_vx]
        else:
            P_vx = get_nan_gradient(P[..., 0], dt, axis = -1)
        if 'v_y' in data_type:
            i_vy = data_type.index('v_y')
            P_vy = P[..., i_vy]
        else:
            P_vy = get_nan_gradient(P[..., 1], dt, axis = -1)
        P_v = np.sqrt(P_vx**2 + P_vy**2)
        P[..., i_v] = P_v
    
    # Do headings, based on previous velocities
    if 'theta' in data_type:
        i_theta = data_type.index('theta')
        if 'v_x' in data_type:
            i_vx = data_type.index('v_x')
            P_vx = P[..., i_vx]
        else:
            P_vx = get_nan_gradient(P[..., 0], dt, axis = -1)
        if 'v_y' in data_type:
            i_vy = data_type.index('v_y')
            P_vy = P[..., i_vy]
        else:
            P_vy = get_nan_gradient(P[..., 1], dt, axis = -1)
        P_theta = np.arctan2(P_vy, P_vx)
        P[..., i_theta] = P_theta

    # Do total accelerations, based on previous velocities
    if 'a' in data_type:
        i_a = data_type.index('a')
        if 'v' in data_type:
            i_v = data_type.index('v')
            P_v = P[..., i_v]
        else:
            i_v = data_type.index('v')
            if 'v_x' in data_type:
                i_vx = data_type.index('v_x')
                P_vx = P[..., i_vx]
            else:
                P_vx = get_nan_gradient(P[..., 0], dt, axis = -1)
            if 'v_y' in data_type:
                i_vy = data_type.index('v_y')
                P_vy = P[..., i_vy]
            else:
                P_vy = get_nan_gradient(P[..., 1], dt, axis = -1)
            P_v = np.sqrt(P_vx**2 + P_vy**2)

        P_a = get_nan_gradient(P_v, dt, axis = -1)
        P[..., i_a] = P_a

    # Do change in heading
    if 'd_theta' in data_type:
        i_d_theta = data_type.index('d_theta')
        if 'theta' in data_type:
            i_theta = data_type.index('theta')
            P_theta = P[..., i_theta]
        else:                
            if 'v_x' in data_type:
                i_vx = data_type.index('v_x')
                P_vx = P[..., i_vx]
            else:
                P_vx = get_nan_gradient(P[..., 0], dt, axis = -1)
            if 'v_y' in data_type:
                i_vy = data_type.index('v_y')
                P_vy = P[..., i_vy]
            else:
                P_vy = get_nan_gradient(P[..., 1], dt, axis = -1)
            P_theta = np.arctan2(P_vy, P_vx)

        P_theta = np.unwrap(P_theta, axis = -1)
        P_d_theta = get_nan_gradient(P_theta, dt, axis = -1)
        P[..., i_d_theta] = P_d_theta

    # Reverse flattening of P
    P = P.reshape(p_shape)
    return P

def get_integrals(X, dt, data_type, control_states):
    # assert control states are in data_type
    for s in control_states:
        assert s in data_type, f"Control state {s} is not in data_type"
    assert X.shape[-1] == len(data_type), "The last dimension of X should correspond to the length of data_type"
    assert data_type[0] == 'x' and data_type[1] == 'y', "The first two elements of data_type should be 'x' and 'y'"

    # Get rules for integrals.
    # We use the following relationships:
    # v_x = gradient(x) / dt
    # v_y = gradient(y) / dt
    # a_x = gradient(v_x) / dt
    # a_y = gradient(v_y) / dt
    # v = sqrt(v_x^2 + v_y^2)
    # theta = atan2(v_y, v_x)
    # a = gradient(v) / dt
    # d_theta = gradient(theta) / dt

    # copy data_type, so we can store intermediate values
    data_type_copy = data_type.copy()

    # I.e, the data has a non-cyclic dependency structure, which allows us to get integrals in a specific order.
    # Go from highest possible control states to lowest.
    if 'a_x' in control_states:
        i_ax = data_type_copy.index('a_x')
        # check if v_x is available
        if 'v_x' in data_type_copy:
            i_vx = data_type_copy.index('v_x')
            X_vx = X[..., i_vx]
        else:
            # v and theta need to be available
            assert 'v' in data_type_copy and 'theta' in data_type_copy, "v and theta need to be available to calculate v_x"
            i_v = data_type_copy.index('v')
            i_theta = data_type_copy.index('theta')
            X_v = X[..., i_v]
            X_theta = X[..., i_theta]
            X_vx = X_v * np.cos(X_theta)
        X_ax = X[..., i_ax]
        # Overwrite v_x based on a_x
        X_vx = get_nan_integral(X_ax, X_vx, dt, axis = -1) 

        if 'v_x' in data_type_copy:
            # Overwrite in X
            X[..., i_vx] = X_vx
            # Add v_x to control state, so that its dependence are also updates
            control_states.append('v_x')
        else:
            # Add v_x to data_type_copy, and append X_vx to X
            data_type_copy.append('v_x')
            X = np.concatenate([X, X_vx[..., np.newaxis]], axis = -1)
            # Add v_x to control state, so that its dependence are also updates
            control_states.append('v_x')
    
    if 'a_y' in control_states:
        i_ay = data_type_copy.index('a_y')
        # check if v_y is available
        if 'v_y' in data_type_copy:
            i_vy = data_type_copy.index('v_y')
            X_vy = X[..., i_vy]
        else:
            # v and theta need to be available
            assert 'v' in data_type_copy and 'theta' in data_type_copy, "v and theta need to be available to calculate v_y"
            i_v = data_type_copy.index('v')
            i_theta = data_type_copy.index('theta')
            X_v = X[..., i_v]
            X_theta = X[..., i_theta]
            X_vy = X_v * np.sin(X_theta)
        X_ay = X[..., i_ay]
        # Overwrite v_y based on a_y
        X_vy = get_nan_integral(X_ay, X_vy, dt, axis = -1) 

        if 'v_y' in data_type_copy:
            # Overwrite in X
            X[..., i_vy] = X_vy
            # Add v_y to control state, so that its dependence are also updates
            control_states.append('v_y')
        else:
            # Add v_y to data_type_copy, and append X_vy to X
            data_type_copy.append('v_y')
            X = np.concatenate([X, X_vy[..., np.newaxis]], axis = -1)
            # Add v_y to control state, so that its dependence are also updates
            control_states.append('v_y')

    if 'd_theta' in control_states:
        i_d_theta = data_type_copy.index('d_theta')
        i_theta = data_type_copy.index('theta') # this needs to be there (is enforced by framework)
        X_d_theta = X[..., i_d_theta]
        X_theta = X[..., i_theta]
        # Overwrite theta based on d_theta
        X_theta = get_nan_integral(X_d_theta, X_theta, dt, axis = -1)

        # Overwrite in X
        X[..., i_theta] = X_theta
        # Add theta to control state, so that its dependence are also updates
        control_states.append('theta')
    
    if 'a' in control_states:
        i_a = data_type_copy.index('a')
        # check if v is available
        if 'v' in data_type_copy:
            i_v = data_type_copy.index('v')
            X_v = X[..., i_v]
        else:
            # v_x and v_y need to be available
            assert 'v_x' in data_type_copy and 'v_y' in data_type_copy, "v_x and v_y need to be available to calculate v"
            i_vx = data_type_copy.index('v_x')
            i_vy = data_type_copy.index('v_y')
            X_vx = X[..., i_vx]
            X_vy = X[..., i_vy]
            X_v = np.sqrt(X_vx**2 + X_vy**2)
        X_a = X[..., i_a]
        # Overwrite v based on a
        X_v = get_nan_integral(X_a, X_v, dt, axis = -1)

        if 'v' in data_type:
            # Overwrite in X
            X[..., i_v] = X_v
            # Add v to control state, so that its dependence are also updates
            control_states.append('v')
        else:
            # Add v to data_type_copy, and append X_v to X
            data_type_copy.append('v')
            X = np.concatenate([X, X_v[..., np.newaxis]], axis = -1)
            # Add v to control state, so that its dependence are also updates
            control_states.append('v')

    if 'v' in control_states:
        # theta should also be a control state, as v depends on it
        assert 'theta' in control_states, "theta should also be a control state, as v depends on it"
        i_v = data_type_copy.index('v')
        i_theta = data_type_copy.index('theta')
        X_v = X[..., i_v]
        X_theta = X[..., i_theta]

        # Calculate v_x and v_y
        X_vx = X_v * np.cos(X_theta)
        X_vy = X_v * np.sin(X_theta)

        # Add or overwrite v_x and v_y in X
        if 'v_x' in data_type_copy:
            i_vx = data_type_copy.index('v_x')
            X[..., i_vx] = X_vx
            control_states.append('v_x')
        else:
            data_type_copy.append('v_x')
            X = np.concatenate([X, X_vx[..., np.newaxis]], axis=-1)
            control_states.append('v_x')
        
        if 'v_y' in data_type_copy:
            i_vy = data_type_copy.index('v_y')
            X[..., i_vy] = X_vy
            control_states.append('v_y')
        else:
            data_type_copy.append('v_y')
            X = np.concatenate([X, X_vy[..., np.newaxis]], axis=-1)
            control_states.append('v_y')
    
    if 'theta' in control_states:
        assert 'v' in control_states, "v should also be a control state, as theta depends on it"
    
    # by now, 'v_x' and 'v_y' should be in the control states, so we can calculate their integrals
    assert 'v_x' in control_states and 'v_y' in control_states, "v_x and v_y should be in the control states, so we can calculate their integrals"
    i_vx = data_type_copy.index('v_x')
    i_vy = data_type_copy.index('v_y')
    X_vx = X[..., i_vx]
    X_vy = X[..., i_vy]
    X_x = X[..., 0]
    X_y = X[..., 1]
    # Overwrite x based on v_x
    X_x = get_nan_integral(X_vx, X_x, dt, axis = -1)
    # Overwrite y based on v_y
    X_y = get_nan_integral(X_vy, X_y, dt, axis = -1)

    # Overwrite in X
    X[..., 0] = X_x
    X[..., 1] = X_y

    # Expand control_states to include positions
    control_states.append('x')
    control_states.append('y')

    # Go through potentially missing control states
    for s in ['a_x', 'a_y', 'v', 'theta', 'a', 'd_theta']:
        if (s not in control_states) and (s in data_type):
            if s == 'a_x':
                i_ax = data_type_copy.index('a_x')
                # v_x should be in control states
                assert 'v_x' in control_states, "v_x should be in control states to calculate a_x"
                i_vx = data_type_copy.index('v_x')
                X_vx = X[..., i_vx]
                # Overwrite v_x based on a_x
                X_ax = get_nan_gradient(X_vx, dt, axis = -1)
                # Overwrite in X
                X[..., i_ax] = X_ax
                # Add a_x to control states
                control_states.append('a_x')
            elif s == 'a_y':
                i_ay = data_type_copy.index('a_y')
                # v_y should be in control states
                assert 'v_y' in control_states, "v_y should be in control states to calculate a_y"
                i_vy = data_type_copy.index('v_y')
                X_vy = X[..., i_vy]
                # Overwrite v_y based on a_y
                X_ay = get_nan_gradient(X_vy, dt, axis = -1)
                # Overwrite in X
                X[..., i_ay] = X_ay
                # Add a_y to control states
                control_states.append('a_y')
            elif s == 'v':
                i_v = data_type_copy.index('v')
                # v_x and v_y should be in control states
                assert 'v_x' in control_states and 'v_y' in control_states, "v_x and v_y should be in control states to calculate v"
                i_vx = data_type_copy.index('v_x')
                i_vy = data_type_copy.index('v_y')
                X_vx = X[..., i_vx]
                X_vy = X[..., i_vy]
                X_v = np.sqrt(X_vx**2 + X_vy**2)
                # Overwrite in X
                X[..., i_v] = X_v
                # Add v to control states
                control_states.append('v')
            elif s == 'theta':
                i_theta = data_type_copy.index('theta')
                # v_x and v_y should be in control states
                assert 'v_x' in control_states and 'v_y' in control_states, "v_x and v_y should be in control states to calculate theta"
                i_vx = data_type_copy.index('v_x')
                i_vy = data_type_copy.index('v_y')
                X_vx = X[..., i_vx]
                X_vy = X[..., i_vy]
                X_theta = np.arctan2(X_vy, X_vx)
                # Overwrite in X
                X[..., i_theta] = X_theta
                # Add theta to control states
                control_states.append('theta')
            elif s == 'a':
                i_a = data_type_copy.index('a')
                # v should be in control states
                assert 'v' in control_states, "v should be in control states to calculate a"
                i_v = data_type_copy.index('v')
                X_v = X[..., i_v]
                # Overwrite v based on a
                X_a = get_nan_gradient(X_v, dt, axis = -1)
                # Overwrite in X
                X[..., i_a] = X_a
                # Add a to control states
                control_states.append('a')
            elif s == 'd_theta':
                i_d_theta = data_type_copy.index('d_theta')
                # theta should be in control states
                assert 'theta' in control_states, "theta should be in control states to calculate d_theta"
                i_theta = data_type_copy.index('theta')
                X_theta = X[..., i_theta]
                # Overwrite theta based on d_theta
                X_theta = np.unwrap(X_theta, axis = -1)
                X_d_theta = get_nan_gradient(X_theta, dt, axis = -1)
                # Overwrite in X
                X[..., i_d_theta] = X_d_theta
                # Add d_theta to control states
                control_states.append('d_theta')


    # Remove added states
    X = X[..., :len(data_type)]
    return X
