import os
import numpy as np
from scipy import sparse
from sys import getsizeof

from joblib import Parallel, delayed, parallel_backend

from ._unot import unot


def cot_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """ Solve the collective optimal transport problem with distance limits.
    
    Parameters
    ----------
    S : (n_pos_s,ns_s) numpy.ndarray
        Source distributions over `n_pos_s` positions of `ns_s` source species.
    D : (n_pos_d,ns_d) numpy.ndarray
        Destination distributions over `n_pos_d` positions of `ns_d` destination species.
    A : (ns_s,ns_d) numpy.ndarray
        The cost coefficients for source-destination species pairs. An infinity value indicates that the two species cannot be coupled.
    M : (n_pos_s,n_pos_d) numpy.ndarray
        The distance (cost) matrix among the positions.
    cutoff : (ns_s,ns_d) numpy.ndarray
        The distance (cost) cutoff between each source-destination species pair. All transports are restricted by the cutoffs.
    eps_p : float, defaults to 1e-1
        The coefficient for entropy regularization of P.
    eps_mu : float, defaults to eps_p
        The coefficient for entropy regularization of unmatched source mass.
    eps_nu : float, defaults to eps_p
        The coefficient for entriopy regularization of unmatched target mass.
    rho : float, defaults to 1e2
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The maximum number of iterations in the unormalized OT problem. Defaults to 1e4.
    stopthr : float, optional
        The relatitive error threshold for terminating the iteration. Defaults to 1e-8.
    
    Returns
    -------
    (ns_s,ns_d,n_pos_s,n_pos_d) numpy.ndarray
        The transport plans among the multiple species.
    """
    np.set_printoptions(precision=2)
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    max_amount = max( S.sum(), D.sum() )
    S = S / max_amount
    D = D / max_amount

    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    # Set up the large collective OT problem
    a = S.flatten('F')
    b = D.flatten('F')
    C = np.inf * np.ones([len(a),len(b)])
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_M = np.array(M)
                tmp_M[np.where(tmp_M > cutoff[i,j])] = np.inf
                C[i*n_pos_s:(i+1)*n_pos_s, j*n_pos_d:(j+1)*n_pos_d] = A[i,j] * tmp_M
    C = C/np.max(C[np.where(~np.isinf(C))])
    nzind_a = np.where(a > 0)[0]
    nzind_b = np.where(b > 0)[0]
    tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
        eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
    P = np.zeros_like(C)
    for i in range(len(nzind_a)):
        for j in range(len(nzind_b)):
            P[nzind_a[i],nzind_b[j]] = tmp_P[i,j]
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for i in range(ns_s):
        for j in range(ns_d):
            P_expand[i,j,:,:] = P[i*n_pos_s:(i+1)*n_pos_s,j*n_pos_d:(j+1)*n_pos_d]
    return P_expand * max_amount


def cot_row_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """Solve for each sender species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for i in range(ns_s):
        a = S[:,i]
        D_ind = np.where(~np.isinf(A[i,:]))[0]
        b = D[:,D_ind].flatten('F')
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C = np.inf * np.ones([len(a), len(b)], float)
        for j in range(len(D_ind)):
            D_j = D_ind[j]
            tmp_M = np.array(M)
            tmp_M[np.where(tmp_M > cutoff[i,D_j])] = np.inf
            C[:,j*n_pos_d:(j+1)*n_pos_d] = A[i,D_j] * tmp_M
        C = C/np.max(C[np.where(~np.isinf(C))])
        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
        P = np.zeros_like(C)
        for ii in range(len(nzind_a)):
            for jj in range(len(nzind_b)):
                P[nzind_a[ii],nzind_b[jj]] = tmp_P[ii,jj]
        for j in range(len(D_ind)):
            P_expand[i,D_ind[j],:,:] = P[:,j*n_pos_d:(j+1)*n_pos_d] * max_amount
    return P_expand


def cot_col_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """Solve for each destination species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for j in range(ns_d):
        b = D[:,j]
        S_ind = np.where(~np.isinf(A[:,j]))[0]
        a = S[:,S_ind].flatten('F')
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C = np.inf * np.ones([len(a), len(b)], float)
        for i in range(len(S_ind)):
            S_i = S_ind[i]
            tmp_M = np.array(M)
            tmp_M[np.where(tmp_M > cutoff[S_i,j])] = np.inf
            C[i*n_pos_s:(i+1)*n_pos_s,:] = A[S_i,j] * tmp_M
        C = C/np.max(C[np.where(~np.isinf(C))])
        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
        P = np.zeros_like(C)
        for ii in range(len(nzind_a)):
            for jj in range(len(nzind_b)):
                P[nzind_a[ii],nzind_b[jj]] = tmp_P[ii,jj]
        for i in range(len(S_ind)):
            P_expand[S_ind[i],j,:,:] = P[i*n_pos_s:(i+1)*n_pos_s,:] * max_amount
    return P_expand


def cot_blk_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """Solve for each pair of species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"
    
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for i in range(ns_s):
        for j in range(ns_d):
            if np.isinf(A[i,j]): continue
            a = S[:,i]; b = D[:,j]
            max_amount = max(a.sum(), b.sum())
            a = a / max_amount; b = b / max_amount
            C = np.array(M)
            C[np.where(C > cutoff[i,j])] = np.inf
            C = C/np.max(C[np.where(~np.isinf(C))])
            nzind_a = np.where(a > 0)[0]
            nzind_b = np.where(b > 0)[0]
            tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
                eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
            P = np.zeros_like(C)
            for ii in range(len(nzind_a)):
                for jj in range(len(nzind_b)):
                    P[nzind_a[ii],nzind_b[jj]] = tmp_P[ii,jj]
            P_expand[i,j,:,:] = P[:,:]
    return P_expand


# ============================================================================
# SPARSE IMPLEMENTATIONS
# ============================================================================

def coo_submatrix_pull(matr, rows, cols):
    """
    Pulls out an arbitrary i.e. non-contiguous submatrix out of
    a sparse.coo_matrix. 
    """
    if type(matr) != sparse.coo_matrix:
        raise TypeError('Matrix must be sparse COOrdinate format')
    
    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1])
    
    lr = len(rows)
    lc = len(cols)
    
    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return sparse.coo_matrix((matr.data[newelem], np.array([gr[newrows],
        gc[newcols]])),(lr, lc))


def cot_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """ Solve the collective optimal transport problem with distance limits in sparse format.
    
    Parameters
    ----------
    S : (n_pos_s,ns_s) numpy.ndarray
        Source distributions over `n_pos_s` positions of `ns_s` source species.
    D : (n_pos_d,ns_d) numpy.ndarray
        Destination distributions over `n_pos_d` positions of `ns_d` destination species.
    A : (ns_s,ns_d) numpy.ndarray
        The cost coefficients for source-destination species pairs. An infinity value indicates that the two species cannot be coupled.
    M : (n_pos_s,n_pos_d) numpy.ndarray
        The distance (cost) matrix among the positions.
    cutoff : (ns_s,ns_d) numpy.ndarray
        The distance (cost) cutoff between each source-destination species pair. All transports are restricted by the cutoffs.
    eps_p : float, defaults to 1e-1
        The coefficient for entropy regularization of P.
    eps_mu : float, defaults to eps_p
        The coefficient for entropy regularization of unmatched source mass.
    eps_nu : float, defaults to eps_p
        The coefficient for entriopy regularization of unmatched target mass.
    rho : float, defaults to 1e2
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The maximum number of iterations in the unormalized OT problem. Defaults to 1e4.
    stopthr : float, optional
        The relatitive error threshold for terminating the iteration. Defaults to 1e-8.
    
    Returns
    -------
    A dictionary of scipy.sparse.coo_matrix
        The transport plan in coo sparse format for source species i and destinaton species j can be retrieved with the key (i,j).
    """
    np.set_printoptions(precision=2)
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    max_amount = max( S.sum(), D.sum() )
    S = S / max_amount
    D = D / max_amount

    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    # Set up the large collective OT problem
    a = S.flatten('F')
    b = D.flatten('F')
    
    C_data, C_row, C_col = [], [], []

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    
    cost_scales = []
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_nzind_s = np.where(S[:,i] > 0)[0]
                tmp_nzind_d = np.where(D[:,j] > 0)[0]
                tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
                tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,j])[0]
                tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
                tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
                cost_scales.append(np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,j])])*A[i,j])
                C_data.append( tmp_M_max_sp.data[tmp_ind]*A[i,j] )
                C_row.append( tmp_row+i*n_pos_s )
                C_col.append( tmp_col+j*n_pos_d )

    cost_scale = np.max(cost_scales)
    C_data = np.concatenate(C_data, axis=0)
    C_row = np.concatenate(C_row, axis=0)
    C_col = np.concatenate(C_col, axis=0)
    C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a),len(b)))

    # Solve the problem on nonzero mass
    nzind_a = np.where(a > 0)[0]
    nzind_b = np.where(b > 0)[0]
    C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

    if verbose:
        print('Number of non-infinity entries in transport cost:', len(C.data))

    del C_data, C_row, C_col, C

    tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
        eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
    
    del C_nz

    P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))
    P = P.tocsr()

    # Output a dictionary of transport plans
    P_expand = {}
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_P = P[i*n_pos_s:(i+1)*n_pos_s, j*n_pos_d:(j+1)*n_pos_d]
                P_expand[(i,j)] = tmp_P.tocoo() * max_amount

    return P_expand    


def cot_row_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """Solve for each sender species separately (sequential version).
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    
    P_expand = {}
    for i in range(ns_s):
        a = S[:,i]
        D_ind = np.where(~np.isinf(A[i,:]))[0]
        b = D[:,D_ind].flatten('F')
        nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
        if len(nzind_a)==0 or len(nzind_b)==0:
            for j in range(len(D_ind)):
                P_expand[(i,D_ind[j])] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s, n_pos_d), dtype=float)
            continue
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for j in range(len(D_ind)):
            D_j = D_ind[j]
            tmp_nzind_s = np.where(S[:,i] > 0)[0]
            tmp_nzind_d = np.where(D[:,D_j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,D_j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append( tmp_M_max_sp.data[tmp_ind]*A[i,D_j] )
            C_row.append( tmp_row )
            C_col.append( tmp_col+j*n_pos_d )
            cost_scales.append( np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,D_j])])*A[i,D_j] )
        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a), len(b)))    

        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

        del C_data, C_row, C_col, C

        tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

        del C_nz

        P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))
        P = P.tocsr()

        for j in range(len(D_ind)):
            tmp_P = P[:,j*n_pos_d:(j+1)*n_pos_d]
            P_expand[(i,D_ind[j])] = tmp_P.tocoo() * max_amount

        del P

    return P_expand


def cot_col_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """Solve for each destination species separately (sequential version).
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    
    P_expand = {}
    for j in range(ns_d):
        S_ind = np.where(~np.isinf(A[:,j]))[0]
        a = S[:,S_ind].flatten('F')
        b = D[:,j]
        nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
        if len(nzind_a)==0 or len(nzind_b)==0:
            for i in range(len(S_ind)):
                P_expand[(S_ind[i],j)] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s,n_pos_d), dtype=float)
            continue
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for i in range(len(S_ind)):
            S_i = S_ind[i]
            tmp_nzind_s = np.where(S[:,S_i] > 0)[0]
            tmp_nzind_d = np.where(D[:,j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[S_i,j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append( tmp_M_max_sp.data[tmp_ind]*A[S_i,j] )
            C_row.append( tmp_row+i*n_pos_s )
            C_col.append( tmp_col )
            cost_scales.append( np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[S_i,j])])*A[S_i,j] )
        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a), len(b)))    

        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

        del C_data, C_row, C_col, C

        tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

        del C_nz

        P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))
        P = P.tocsr()

        for i in range(len(S_ind)):
            tmp_P = P[i*n_pos_s:(i+1)*n_pos_s,:]
            P_expand[(S_ind[i],j)] = tmp_P.tocoo() * max_amount

        del P

    return P_expand


def cot_blk_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """Solve for each pair of species separately (sequential version).
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu), abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"
    
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)

    P_expand = {}
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                a = S[:,i]; b = D[:,j]
                nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
                if len(nzind_a)==0 or len(nzind_b)==0:
                    P_expand[(i,j)] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s, n_pos_d), dtype=float)
                    continue
                max_amount = max(a.sum(), b.sum())
                a = a / max_amount; b = b / max_amount
                tmp_nzind_s = np.where(S[:,i] > 0)[0]
                tmp_nzind_d = np.where(D[:,j] > 0)[0]
                tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
                tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,j])[0]
                tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
                tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
                C_data = tmp_M_max_sp.data[tmp_ind] * A[i,j]
                cost_scale = np.max( M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,j])] )*A[i,j]
                C = sparse.coo_matrix((C_data/cost_scale, (tmp_row, tmp_col)), shape=(len(a), len(b)))

                nzind_a = np.where(a > 0)[0]
                nzind_b = np.where(b > 0)[0]
                C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

                del C_data, C

                tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
                    eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

                del C_nz

                P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))

                P_expand[(i,j)] = P * max_amount
    
    return P_expand


# ============================================================================
# PARALLEL IMPLEMENTATIONS
# ============================================================================

def cot_row_sparse_parallel(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None,
                            rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False, n_jobs=-1):
    """Parallel version of cot_row_sparse — each sender species computed independently.

    Uses process-based parallelism (loky) to avoid GIL contention. Each worker
    process solves one sender species' OT problem independently. Large numpy
    arrays (M, S, D) are automatically memory-mapped by joblib to avoid
    duplication.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p - eps_mu), abs(eps_p - eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    # Pre-compute shared sparse distance matrix
    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row, M_col], (M_row, M_col)), shape=M.shape)

    def _compute_row(i):
        """Compute OT for a single sender species i vs all coupled receivers."""
        a = S[:, i]
        D_ind = np.where(~np.isinf(A[i, :]))[0]
        b = D[:, D_ind].flatten('F')
        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]

        results = {}
        if len(nzind_a) == 0 or len(nzind_b) == 0:
            for j in range(len(D_ind)):
                results[(i, D_ind[j])] = sparse.coo_matrix(
                    ([], ([], [])), shape=(n_pos_s, n_pos_d), dtype=float)
            return results

        max_amount = max(a.sum(), b.sum())
        a_norm = a / max_amount
        b_norm = b / max_amount

        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for j in range(len(D_ind)):
            D_j = D_ind[j]
            tmp_nzind_s = np.where(S[:, i] > 0)[0]
            tmp_nzind_d = np.where(D[:, D_j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i, D_j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append(tmp_M_max_sp.data[tmp_ind] * A[i, D_j])
            C_row.append(tmp_row)
            C_col.append(tmp_col + j * n_pos_d)
            cost_scales.append(
                np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i, D_j])]) * A[i, D_j])

        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix(
            (C_data / cost_scale, (C_row, C_col)), shape=(len(a_norm), len(b_norm)))

        nzind_a = np.where(a_norm > 0)[0]
        nzind_b = np.where(b_norm > 0)[0]
        C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

        tmp_P = unot(a_norm[nzind_a], b_norm[nzind_b], C_nz, eps_p, rho,
                     eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True,
                     solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

        P = sparse.coo_matrix(
            (tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])),
            shape=(len(a_norm), len(b_norm)))
        P = P.tocsr()

        for j in range(len(D_ind)):
            tmp_P = P[:, j * n_pos_d:(j + 1) * n_pos_d]
            results[(i, D_ind[j])] = tmp_P.tocoo() * max_amount

        return results

    # Run in parallel using process-based backend (avoids GIL)
    n_cores = n_jobs if n_jobs > 0 else os.cpu_count()
    if n_cores > 1 and ns_s > 1:
        all_results = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
            delayed(_compute_row)(i) for i in range(ns_s))
    else:
        all_results = [_compute_row(i) for i in range(ns_s)]

    P_expand = {}
    for res in all_results:
        P_expand.update(res)
    return P_expand


def cot_col_sparse_parallel(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None,
                            rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False, n_jobs=-1):
    """Parallel version of cot_col_sparse — each receiver species computed independently.

    Uses process-based parallelism (loky) to avoid GIL contention.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p - eps_mu), abs(eps_p - eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    # Pre-compute shared sparse distance matrix
    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row, M_col], (M_row, M_col)), shape=M.shape)

    def _compute_col(j):
        """Compute OT for a single receiver species j vs all coupled senders."""
        S_ind = np.where(~np.isinf(A[:, j]))[0]
        a = S[:, S_ind].flatten('F')
        b = D[:, j]
        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]

        results = {}
        if len(nzind_a) == 0 or len(nzind_b) == 0:
            for i in range(len(S_ind)):
                results[(S_ind[i], j)] = sparse.coo_matrix(
                    ([], ([], [])), shape=(n_pos_s, n_pos_d), dtype=float)
            return results

        max_amount = max(a.sum(), b.sum())
        a_norm = a / max_amount
        b_norm = b / max_amount

        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for i in range(len(S_ind)):
            S_i = S_ind[i]
            tmp_nzind_s = np.where(S[:, S_i] > 0)[0]
            tmp_nzind_d = np.where(D[:, j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[S_i, j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append(tmp_M_max_sp.data[tmp_ind] * A[S_i, j])
            C_row.append(tmp_row + i * n_pos_s)
            C_col.append(tmp_col)
            cost_scales.append(
                np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[S_i, j])]) * A[S_i, j])

        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix(
            (C_data / cost_scale, (C_row, C_col)), shape=(len(a_norm), len(b_norm)))

        nzind_a = np.where(a_norm > 0)[0]
        nzind_b = np.where(b_norm > 0)[0]
        C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

        tmp_P = unot(a_norm[nzind_a], b_norm[nzind_b], C_nz, eps_p, rho,
                     eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True,
                     solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

        P = sparse.coo_matrix(
            (tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])),
            shape=(len(a_norm), len(b_norm)))
        P = P.tocsr()

        for i in range(len(S_ind)):
            tmp_P = P[i * n_pos_s:(i + 1) * n_pos_s, :]
            results[(S_ind[i], j)] = tmp_P.tocoo() * max_amount

        return results

    # Run in parallel using process-based backend (avoids GIL)
    n_cores = n_jobs if n_jobs > 0 else os.cpu_count()
    if n_cores > 1 and ns_d > 1:
        all_results = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
            delayed(_compute_col)(j) for j in range(ns_d))
    else:
        all_results = [_compute_col(j) for j in range(ns_d)]

    P_expand = {}
    for res in all_results:
        P_expand.update(res)
    return P_expand


def cot_blk_sparse_parallel(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None,
                            rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False, n_jobs=-1):
    """Parallel version of cot_blk_sparse — each (i,j) pair computed independently.

    Uses process-based parallelism (loky) to avoid GIL contention.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p - eps_mu), abs(eps_p - eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row, M_col], (M_row, M_col)), shape=M.shape)

    # Collect all (i,j) pairs to compute
    tasks = []
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i, j]):
                tasks.append((i, j))

    n_cores = n_jobs if n_jobs > 0 else os.cpu_count()
    if verbose:
        print(f'  Parallel COT_BLK: {len(tasks)} L-R pairs on {n_cores} processes')

    def _compute_single_pair(i, j):
        """Compute OT for a single L-R pair."""
        a = S[:, i]; b = D[:, j]
        nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]

        if len(nzind_a) == 0 or len(nzind_b) == 0:
            return (i, j), sparse.coo_matrix(([], ([], [])), shape=(n_pos_s, n_pos_d), dtype=float)

        max_amount = max(a.sum(), b.sum())
        a_norm = a / max_amount
        b_norm = b / max_amount

        tmp_nzind_s = np.where(S[:, i] > 0)[0]
        tmp_nzind_d = np.where(D[:, j] > 0)[0]
        tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
        tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i, j])[0]
        tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
        tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]

        C_data = tmp_M_max_sp.data[tmp_ind] * A[i, j]
        cost_scale = np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i, j])]) * A[i, j]
        C_local = sparse.coo_matrix((C_data / cost_scale, (tmp_row, tmp_col)), shape=(len(a), len(b)))

        nzind_a_local = np.where(a_norm > 0)[0]
        nzind_b_local = np.where(b_norm > 0)[0]
        C_nz = coo_submatrix_pull(C_local, nzind_a_local, nzind_b_local)

        tmp_P = unot(a_norm[nzind_a_local], b_norm[nzind_b_local], C_nz, eps_p, rho,
                     eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver,
                     nitermax=nitermax, stopthr=stopthr)

        P = sparse.coo_matrix((tmp_P.data, (nzind_a_local[tmp_P.row], nzind_b_local[tmp_P.col])),
                              shape=(len(a), len(b)))

        return (i, j), P * max_amount

    # Parallel computation using process-based backend (avoids GIL)
    if n_cores > 1 and len(tasks) > 1:
        results = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
            delayed(_compute_single_pair)(i, j) for i, j in tasks)
    else:
        results = [_compute_single_pair(i, j) for i, j in tasks]

    P_expand = dict(results)
    return P_expand


# ============================================================================
# COMBINED COT — CONCURRENT VARIANT EXECUTION
# ============================================================================

def cot_combine_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None,
                       rho=1e1, weights=(0.25, 0.25, 0.25, 0.25), nitermax=1e4,
                       stopthr=1e-8, verbose=False, n_jobs=-1):
    """Solve the collective optimal transport by combining four variant strategies.

    Runs the four COT variants sequentially, but parallelizes the inner loops
    of cot_row, cot_col, and cot_blk using process-based parallelism (loky).
    Each variant gets the full process budget when it runs.

    Parameters
    ----------
    S : (n_pos_s, ns_s) numpy.ndarray
        Source distributions.
    D : (n_pos_d, ns_d) numpy.ndarray
        Destination distributions.
    A : (ns_s, ns_d) numpy.ndarray
        Cost coefficients. Infinity = uncoupled.
    M : (n_pos_s, n_pos_d) numpy.ndarray
        Distance matrix.
    cutoff : (ns_s, ns_d) numpy.ndarray
        Distance cutoffs per species pair.
    eps_p, eps_mu, eps_nu, rho : float or tuple
        Regularization parameters. If tuple, one per variant (cot, row, col, blk).
    weights : tuple of 4 floats
        Combination weights for the four variants (must sum to 1).
    nitermax : int
        Maximum Sinkhorn iterations.
    stopthr : float
        Convergence threshold.
    verbose : bool
        Print progress information.
    n_jobs : int, default=-1
        Number of parallel worker processes for row/col/blk variants.
        -1 uses all available cores. Set to 1 for fully sequential execution.

    Returns
    -------
    dict of scipy.sparse.coo_matrix
        Combined transport plan for each coupled (i, j) pair.
    """
    # Unpack per-variant parameters (support tuple or scalar)
    if isinstance(eps_p, tuple):
        eps_p_cot, eps_p_row, eps_p_col, eps_p_blk = eps_p
    else:
        eps_p_cot = eps_p_row = eps_p_col = eps_p_blk = eps_p

    if isinstance(rho, tuple):
        rho_cot, rho_row, rho_col, rho_blk = rho
    else:
        rho_cot = rho_row = rho_col = rho_blk = rho

    if eps_mu is None:
        eps_mu_cot = eps_p_cot; eps_mu_row = eps_p_row
        eps_mu_col = eps_p_col; eps_mu_blk = eps_p_blk
    elif isinstance(eps_mu, tuple):
        eps_mu_cot, eps_mu_row, eps_mu_col, eps_mu_blk = eps_mu
    else:
        eps_mu_cot = eps_mu_row = eps_mu_col = eps_mu_blk = eps_mu

    if eps_nu is None:
        eps_nu_cot = eps_p_cot; eps_nu_row = eps_p_row
        eps_nu_col = eps_p_col; eps_nu_blk = eps_p_blk
    elif isinstance(eps_nu, tuple):
        eps_nu_cot, eps_nu_row, eps_nu_col, eps_nu_blk = eps_nu
    else:
        eps_nu_cot = eps_nu_row = eps_nu_col = eps_nu_blk = eps_nu

    n_total = n_jobs if n_jobs > 0 else os.cpu_count()

    if verbose:
        print(f'  COT_COMBINE: {n_total} worker processes, sequential variant execution')

    # Phase 1: cot_sparse — always sequential (one big combined OT problem)
    if verbose:
        print(f'    Phase 1/4: cot_sparse (sequential)')
    P_cot = cot_sparse(S, D, A, M, cutoff,
                       eps_p=eps_p_cot, eps_mu=eps_mu_cot, eps_nu=eps_nu_cot,
                       rho=rho_cot, nitermax=nitermax, stopthr=stopthr, verbose=False)

    # Phase 2: cot_row — parallel across sender species
    if verbose:
        print(f'    Phase 2/4: cot_row_sparse_parallel ({n_total} processes)')
    P_row = cot_row_sparse_parallel(S, D, A, M, cutoff,
                                    eps_p=eps_p_row, eps_mu=eps_mu_row, eps_nu=eps_nu_row,
                                    rho=rho_row, nitermax=nitermax, stopthr=stopthr,
                                    verbose=False, n_jobs=n_total)

    # Phase 3: cot_col — parallel across receiver species
    if verbose:
        print(f'    Phase 3/4: cot_col_sparse_parallel ({n_total} processes)')
    P_col = cot_col_sparse_parallel(S, D, A, M, cutoff,
                                    eps_p=eps_p_col, eps_mu=eps_mu_col, eps_nu=eps_nu_col,
                                    rho=rho_col, nitermax=nitermax, stopthr=stopthr,
                                    verbose=False, n_jobs=n_total)

    # Phase 4: cot_blk — parallel across (ligand, receptor) pairs
    if verbose:
        print(f'    Phase 4/4: cot_blk_sparse_parallel ({n_total} processes)')
    P_blk = cot_blk_sparse_parallel(S, D, A, M, cutoff,
                                    eps_p=eps_p_blk, eps_mu=eps_mu_blk, eps_nu=eps_nu_blk,
                                    rho=rho_blk, nitermax=nitermax, stopthr=stopthr,
                                    verbose=False, n_jobs=n_total)

    # Combine the four variants with specified weights
    P = {}
    w_cot, w_row, w_col, w_blk = float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not np.isinf(A[i, j]):
                P[(i, j)] = (w_cot * P_cot[(i, j)] + w_row * P_row[(i, j)]
                             + w_col * P_col[(i, j)] + w_blk * P_blk[(i, j)])
    return P