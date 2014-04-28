import numpy as np, cython
cimport numpy as np

from scipy.special import ndtr, ndtri

"""
This module has a code to sample from a truncated normal distribution
specified by a set of affine constraints.
"""

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_truncnorm_white(np.ndarray[DTYPE_float_t, ndim=2] A, 
                           np.ndarray[DTYPE_float_t, ndim=1] b, 
                           np.ndarray[DTYPE_float_t, ndim=1] initial, 
                           DTYPE_float_t sigma=1.,
                           DTYPE_int_t burnin=500,
                           DTYPE_int_t ndraw=1000,
                           ):
    """
    Sample from a truncated normal with covariance
    equal to sigma**2 I.

    Constraint is $Ax \leq b$ where `A` has shape
    `(q,n)` with `q` the number of constraints and
    `n` the number of random variables.


    Parameters
    ----------

    A : np.float((q,n))
        Linear part of affine constraints.

    b : np.float(q)
        Offset part of affine constraints.

    initial : np.float(n)
        Initial point for Gibbs draws.
        Assumed to satisfy the constraints.

    sigma : float
        Variance parameter.

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    Returns
    -------

    trunc_sample : np.float((ndraw, n))

    """

    cdef int nvar = A.shape[1]
    cdef int nconstraint = A.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=2] trunc_sample = \
            np.empty((ndraw, nvar), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] state = initial.copy()
    cdef int idx, iter_count, irow, ivar
    cdef double lower_bound, upper_bound, V
    cdef double cdfL, cdfU, unif, tnorm, val, alpha

    cdef double tol = 1.e-7

    cdef np.ndarray[DTYPE_float_t, ndim=1] U = np.dot(A, state) - b

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw)

    # directions not parallel to coordinate axes
    # NOT BEING USED CURRENTLY
    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack([A, 
                   np.random.standard_normal((int(nvar/5),nvar))])

    directions /= np.sqrt((directions**2).sum(1))[:,None]

    cdef int ndir = directions.shape[0]

    cdef np.ndarray[DTYPE_float_t, ndim=2] alphas_dir = \
        np.dot(A, directions.T)

    cdef np.ndarray[DTYPE_float_t, ndim=2] alphas_coord = A
        
    cdef np.ndarray[DTYPE_float_t, ndim=1] alphas_max_dir = \
        np.fabs(alphas_dir).max(0) * tol    

    cdef np.ndarray[DTYPE_float_t, ndim=1] alphas_max_coord = \
        np.fabs(alphas_coord).max(0) * tol 

    # choose the order of sampling (randomly)

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_dir = \
        np.random.random_integers(0, ndir-1, size=(burnin+ndraw,))

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_coord = \
        np.random.random_integers(0, nvar-1, size=(burnin+ndraw,))

    # for switching between coordinate updates and
    # other directions

    cdef int invperiod = 20
    cdef int docoord = 0
    cdef int iperiod = 0

    for iter_count in range(ndraw + burnin):

        iperiod = iperiod + 1
        if iperiod == invperiod:
            docoord = 0
            iperiod = 0
        else:
            docoord = 1

        docoord = 1 # other directions
                    # is buggy
        if docoord == 1:
            idx = random_idx_coord[iter_count]
            V = state[idx]
        else:
            idx = random_idx_dir[iter_count]
            V = 0
            for ivar in range(nvar):
                V = V + directions[idx, ivar] * state[ivar]

        lower_bound = -1e12
        upper_bound = 1e12
        for irow in range(nconstraint):
            if docoord == 1:
                alpha = alphas_coord[irow,idx]
                val = -U[irow] / alpha + V
                if alpha > alphas_max_coord[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_coord[idx] and (val > lower_bound):
                    lower_bound = val
            else:
                alpha = alphas_dir[irow,idx]
                val = -U[irow] / alpha + V
                if alpha > alphas_max_dir[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_dir[idx] and (val > lower_bound):
                    lower_bound = val
        if lower_bound > V:
            lower_bound = V - tol * sigma
        elif upper_bound < V:
            upper_bound = V + tol * sigma

        lower_bound = lower_bound / sigma
        upper_bound = upper_bound / sigma

        if lower_bound < 0:
            cdfL = ndtr(lower_bound)
            cdfU = ndtr(upper_bound)
            unif = usample[iter_count] * (cdfU - cdfL) + cdfL
            if unif < 0.5:
                tnorm = ndtri(unif) * sigma
            else:
                tnorm = -ndtri(1-unif) * sigma
        else:
            cdfL = ndtr(-lower_bound)
            cdfU = ndtr(-upper_bound)
            unif = usample[iter_count] * (cdfL - cdfU) + cdfU
            if unif < 0.5:
                tnorm = -ndtri(unif) * sigma
            else:
                tnorm = ndtri(1-unif) * sigma
            
        if docoord == 1:
            state[idx] = tnorm
            tnorm = tnorm - V
            for irow in range(nconstraint):
                U[irow] = U[irow] + tnorm * A[irow, idx]
        else:
            tnorm = tnorm - V
            for ivar in range(nvar):
                state[ivar] = state[ivar] + tnorm * directions[ivar,idx]
            for irow in range(nconstraint):
                U[irow] = (U[irow] + A[irow, ivar] * 
                           tnorm * directions[ivar,idx])

        if iter_count >= burnin:
            for ivar in range(nvar):
                trunc_sample[iter_count - burnin, ivar] = state[ivar]
        
    return trunc_sample

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_truncnorm_white_sphere(np.ndarray[DTYPE_float_t, ndim=2] A, 
                                  np.ndarray[DTYPE_float_t, ndim=1] b, 
                                  np.ndarray[DTYPE_float_t, ndim=1] initial, 
                                  DTYPE_int_t burnin=500,
                                  DTYPE_int_t ndraw=1000,
                                  ):
    """
    Sample from a truncated normal with covariance
    equal to I restricted to a sphere of 
    radius `np.linalg.norm(initial)`.

    Constraint is $Ax \leq b$ where `A` has shape
    `(q,n)` with `q` the number of constraints and
    `n` the number of random variables.


    Parameters
    ----------

    A : np.float((q,n))
        Linear part of affine constraints.

    b : np.float(q)
        Offset part of affine constraints.

    initial : np.float(n)
        Initial point for Gibbs draws.
        Assumed to satisfy the constraints.

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    Returns
    -------

    trunc_sample : np.float((ndraw, n))

    """

    cdef int nvar = A.shape[1]
    cdef int nconstraint = A.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=2] trunc_sample = \
            np.empty((ndraw, nvar), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] state = initial.copy()
    cdef int idx, iter_count, irow, ivar
    cdef double lower_bound, upper_bound, V
    cdef double U, L
    cdef double norm_state = np.linalg.norm(state)
    cdef double tol = 1.e-7

    cdef np.ndarray[DTYPE_float_t, ndim=1] Astate = np.dot(A, state)

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw)

    # directions not parallel to coordinate axes
    # NOT BEING USED CURRENTLY
    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack([np.identity(nvar),
                   A, 
                   np.random.standard_normal((int(nvar/5),nvar))])

    directions /= np.sqrt((directions**2).sum(1))[:,None]

    cdef int ndir = directions.shape[0]

    cdef np.ndarray[DTYPE_float_t, ndim=2] Adir = \
        np.dot(A, directions.T)

    cdef double theta, cos_theta, sin_theta_norm, dir_state

    # choose the order of sampling (randomly)

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_dir = \
        np.random.random_integers(0, ndir-1, size=(burnin+ndraw,))

    for iter_count in range(ndraw + burnin):

        lower_bound = -np.pi
        upper_bound = np.pi

        eta = directions[idx] - (directions[idx] * state).sum() * state / (norm_state**2)
        eta = eta / np.linalg.norm(eta)

        for irow in range(nconstraint):
            a1 = Astate[irow]
            a2 = 0
            for ivar in range(nvar):
                a2 = a2 + A[irow,ivar] * eta[ivar]
            L, U = _find_interval(a1, a2 * norm_state, b[irow])
            if L != -np.pi:
                print (a1*np.cos(L) + a2*norm_state*np.sin(L) - b[irow], 
                       a1*np.cos(U) + a2*norm_state*np.sin(U) - b[irow],
                       a1*np.cos((L+U)/2) + a2*norm_state*np.sin((L+U)/2) - b[irow]), 'soln'
                if ((a1*np.cos((L+U)/2) + a2*norm_state*np.sin((L+U)/2) - b[irow]) > 0):
                    raise ValueError(`a1,a2*norm_state,b[irow],L,U`)
            print L, U, a1, b[irow], (a1 <= b[irow])
            if L > lower_bound:
                lower_bound = L
            if U < upper_bound:
                upper_bound = U

        theta = lower_bound + usample[iter_count] * (upper_bound - lower_bound)
        cos_theta = np.cos(theta)
        sin_theta_norm = np.sin(theta) * norm_state

        print 'before step', (np.dot(A, state) - b).max()

        for ivar in range(nvar):
            state[ivar] = cos_theta * state[ivar] + sin_theta_norm * eta[ivar]

        state_new = cos_theta * state + sin_theta_norm * eta
        print np.linalg.norm(state_new), np.linalg.norm(state), theta, lower_bound, upper_bound, np.linalg.norm(eta)
        for irow in range(nconstraint):
            Astate[irow] = 0
            for ivar in range(nvar):
                Astate[irow] = Astate[irow] + A[irow,ivar] * state[ivar]

        print 'step taken', (np.dot(A, state) - b).max()

        if iter_count >= burnin:
            for ivar in range(nvar):
                trunc_sample[iter_count - burnin, ivar] = state[ivar]
        
    return trunc_sample

def _find_interval(a1, a2, b):
    """
    Find the interval 

    {t: a1*cos(t) + a2*sin(t) <= b}

    under the assumption that a1 <= b (i.e. the interval is non-empty).

    The assumption is not checked.

    """
    norm_a = np.sqrt(a1**2+a2**2)
    if np.fabs(b / norm_a) < 1:
        alpha = np.arcsin(a1/norm_a)
        if a2 < 0:
            alpha = np.pi - alpha
        tstar1 = np.arcsin(b/norm_a) - alpha
        if tstar1 > np.pi:
            tstar1 = tstar1 - 2 * np.pi
        tstar2 = (np.pi - np.arcsin(b/norm_a) - alpha) % (2 * np.pi)
        if tstar2 > np.pi:
            tstar2 = tstar2 - 2 * np.pi
        lower, upper = sorted([tstar1, tstar2])
        return lower, upper
    else:
        return -np.pi, np.pi

