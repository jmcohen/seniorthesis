#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

import numpy as np 
from math import sqrt
from myfista import mfista
import scipy
from scipy.linalg.interpolative import estimate_spectral_norm
from scipy import sparse

""" 
	This library contains functions to solve the regression problem:

	    argmin  (1/2) || Y - W M ||^2_F + omega(M)
	      M

	for various penalty functions omega(M).

	The functions are:
	   regress_ols
	   regress_ridge
	   regress_ortho
	   regress_elastic_net
	   regress_smooth_lasso
	   regress_spectral_lasso
	   regress_wedge

	Dimensions:   Y is [nframes x nvoxels]
	              W is [nframes x ncomponents]
	              M is [ncomponents x nvoxels]

"""

def ols_loss(Y, W, M):
	"""  Loss function for OLS regression: 

	 1/2 ||Y - WM||^2_F

	 Parameters
	 ----------
	 Y : ndarray, shape (nframes, nvoxels)
	 	subject data
	 W : ndarray, shape (nframes, ncomponents)
	 	timecourses

	 """
	return squared_loss(Y, W, M)

def regress_ols(Y, W):
	""" OLS regression

	omega(M) = 0

	Parameters
	 ----------
	Y : ndarray, shape (nframes, nvoxels)
	 	subject data
	W : ndarray, shape (nframes, ncomponents)
	 	timecourses

	Returns
	-------
	M : spatial maps
		ndarray, shape (ncomponents, nvoxels)

	"""
	return np.linalg.solve(W.T.dot(W), W.T.dot(Y))

def ridge_loss(Y, W, M, beta):
	"""  Loss function for ridge regression: 

	 1/2 ||Y - WM||^2_F + beta * ||M||^2_F

	Parameters
	----------
	Y : ndarray, shape (nframes, nvoxels)
	 	subject data
	W : ndarray, shape (nframes, ncomponents)
	 	timecourses
	beta : float
		strength of l2 penalty

	Returns
	-------
	float

	"""
	return squared_loss(Y, W, M) + beta * l2(M)

def regress_ridge(Y, W, beta):
	""" omega(M) = beta * ||M||^2_F 

	Parameters
	---------
	Y : ndarray, shape (nframes, nvoxels)
	 	subject data
	W : ndarray, shape (nframes, ncomponents)
	 	timecourses
	beta : float
		strength of l2 penalty

	Returns
	-------
	M : spatial maps
		ndarray, shape (ncomponents, nvoxels)

	"""
	ncomponents = W.shape[1]
	return np.linalg.solve(W.T.dot(W) + np.eye(ncomponents) * beta, W.T.dot(Y))

def ortho_loss(Y, W, M):
	""" Loss function for orthogonal regression:

	1/2 ||Y - WM||^2_F

	Parameters
	---------
	Y : ndarray, shape (nframes, nvoxels)
		subject data
	W : ndarray, shape (nframes, ncomponents)
		timecourses
		
	Returns
	-------
	float

	"""
	return squared_loss(Y, W, M)

def regress_ortho(Y, W):
	""" Orthogonal regression

	argmin  ||Y - W M||^2_F  subject to   M M^T = I
	  M 

	Parameters
	---------
	Y : ndarray, shape (nframes, nvoxels)
		subject data
	W : ndarray, shape (nframes, ncomponents)
		timecourses

	Returns
	-------
	M : ndarray, shape (ncomponents, nvoxels)
		spatial maps

	"""

	nframes, ncomponents = W.shape
	Um, sm, Vm = np.linalg.svd(Y.T.dot(W), full_matrices=False)
	return (Um.dot(Vm)).T

def regress_ortho_scaled(Y, W):
	""" argmin  ||Y - W M||^2_F such that M M^T = c*I  for some c 
	     M,c

	This function optimizes over *c* too -- that is, it finds the c such that
	all spatial maps are orthogonal 


	That is, each spatial map has the same norm and the spatial maps
	are pairwise orthogonal

	Parameters
	---------
	Y : ndarray, shape (nframes, nvoxels)
		subject data
	W : ndarray, shape (nframes, ncomponents)
		timecourses

	Returns
	-------
	M : spatial maps
		ndarray, shape (ncomponents, nvoxels)

	"""

	nframes, ncomponents = W.shape
	Um, sm, VmT = np.linalg.svd(Y.T.dot(W), full_matrices=False)
	return (sm.sum() / np.linalg.norm(W, 'fro')) * (Um.dot(VmT)).T

def hthresh(C, lam):
	""" hard threshold C at lam """

	M = C.copy()
	M[np.abs(M) <= lam] = 0
	return M

def elastic_net_loss(Y, W, M, alpha, beta):
	"""  Loss function for elastic net regression: 

	 1/2 ||Y - WM||^2_F + alpha * ||M||_1  + beta * ||M||^2_F

	 Parameters
	 ----------
	 Y : ndarray, shape (nframes, nvoxels)
	 	subject cata
	 W : ndarray, shape (nframes, ncomponents)
	 	timecourses
	 alpha : float
	 	strength of l1 penalty
	 beta : float
	 	strength of l2 penalty 

	Returns
	-------
	float 

	 """
	return squared_loss(Y, W, M) + alpha*l1(M) + beta*l2(M) 

def regress_elastic_net(Y, W, alpha, beta, step_size_factor=100.0):
	""" Elastic net regression

	omega(M) = alpha * ||M||_1 + beta * ||M||^2_F 

	Parameters
	---------
	Y : ndarray, shape (nframes, nvoxels)
	 	subject cata
	W : ndarray, shape (nframes, ncomponents)
	 	timecourses
	alpha : float
		strength of l1 penalty
	beta : float
		strength of l2 penalty 

	Returns
	-------
	M : ndarray, shape (ncomponents, nvoxels)
		spatial maps

	"""
	nframes, nvoxels = Y.shape
	ncomponents = W.shape[1]

	WtW = W.T.dot(W)
	WtY = W.T.dot(Y)

	def f1(m):
		M = m.reshape((ncomponents, nvoxels))
		return squared_loss(Y, W, M) + beta*l2(M)

	def f1_grad(m):
		M = m.reshape((ncomponents, nvoxels))
		gradient = squared_loss_gradient(WtW, WtY, M) + beta*l2_gradient(M)
		return gradient.ravel()

	def f2(m):
		return alpha*l1(m)

	def f2_prox(m, l):
		return prox_l1(m, alpha * l)

	def total_energy(m):
		return f1(m) + f2(m)

   	# lipschitz constant of f1_grad
   	"""
   	todo todo todo check this lipschitz constant, i think it's wrong
   	"""
   	lipschitz = sqrt(nvoxels)*np.linalg.norm(WtW, 2) + 2*beta*sqrt(ncomponents)
	lipschitz /= step_size_factor

	m, history, _ = mfista(f1_grad, f2_prox, total_energy,
	    lipschitz*1.1, ncomponents*nvoxels, max_iter=10000, tol=1.0,
		check_lipschitz=False, callback=None, verbose=1)

	niter = len(history)

	energy_after = total_energy(m)
	M = m.reshape((ncomponents, nvoxels))

	print "\t%d iters" % niter
	print "\t%f percent sparse" % pct_sparse(M)
	
	return M


""" TODO TODO: i really do need to verify the lipschitz bound for the below function """

def smooth_lasso_loss(Y, W, M, alpha, beta, gamma, laplacian):
	""" Loss function for smooth lasso regression

	||Y - W*M||^2_F + alpha * ||M||_1 + beta * ||M||^2_F + gamma * \sum_k (m_k)^T L m_k 

	Parameters
	----------
	Y : ndarray, shape (nframes, nvoxels)
	 	subject cata
	W : ndarray, shape (nframes, ncomponents)
	 	timecourses
	alpha : float
		strength of l1 penalty
	beta : float
		strength of l2 penalty 
	gamma : float
		strength of smoothness penalty

	Returns
	-------
	float

	"""
	return squared_loss(Y, W, M) + alpha*l1(M) + beta*l2(M) + gamma*smoothness(M, laplacian)

def regress_smooth_lasso(Y, W, alpha, beta, gamma, laplacian, step_size_factor=100.0):
	""" Smooth lasso regression

	omega(M) = alpha * ||M||_1 + beta * ||M||^2_F + gamma * \sum_k (m_k)^T L m_k 

	Parameters
	----------
	Y : ndarray, shape (nframes, nvoxels)
	 	subject cata
	W : ndarray, shape (nframes, ncomponents)
	 	timecourses
	alpha : float
		strength of l1 penalty
	beta : float
		strength of l2 penalty 
	gamma : float
		strength of smoothness penalty (l2 penalty on difference between neighboring voxels)
	laplacian : sparse ndarray, shape (nvoxels, nvoxels)
		the graph laplacian for the voxel grid
	step_size_factor : float
		pretend that the Lipschitz constant of the objective function is smaller than it really is by this factor.
		step_size_factor > 1 will speed up the optimization, but may lead to errors

	Returns
	-------
	M : spatial maps
		ndarray, shape (ncomponents, nvoxels)

	"""
	nframes, nvoxels = Y.shape
	ncomponents = W.shape[1]

	WtW = W.T.dot(W)
	WtY = W.T.dot(Y)

	def f1(m):
		M = m.reshape((ncomponents, nvoxels))
		return squared_loss(Y, W, M) + beta*l2(M) + gamma*smoothness(M, laplacian)

	def f1_grad(m):
		M = m.reshape((ncomponents, nvoxels))
		gradient = squared_loss_gradient(WtW, WtY, M) + beta*l2_gradient(M) + gamma*smoothness_gradient(M, laplacian)
		return gradient.ravel()

	def f2(m):
		return alpha*l1(m)

	def f2_prox(m, l):
		return prox_l1(m, alpha * l)

	def total_energy(m):
		return f1(m) + f2(m)

   	laplacian_spectral_norm = estimate_spectral_norm(laplacian, 200)

   	# lipschitz constant of f1_grad
   	lipschitz = sqrt(nvoxels)*np.linalg.norm(W, 2) + 2*gamma*sqrt(ncomponents)*laplacian_spectral_norm + 2*beta*sqrt(ncomponents)
	lipschitz /= step_size_factor

	m, history, _ = mfista(f1_grad, f2_prox, total_energy,
	    lipschitz*1.1, ncomponents*nvoxels, max_iter=10000, tol=1.0,
		check_lipschitz=False, callback=None, verbose=1)

	niter = len(history)

	energy_after = total_energy(m)
	M = m.reshape((ncomponents, nvoxels))

	print "\t%d iters" % niter
	print "\t%f percent sparse" % pct_sparse(M)

	return M


def l1(M):
	""" Computes the elementwise l1 penalty:

	   || M ||_1

	Parameters
	----------
	M : ndarray, shape (ncomponents, nvoxels)

	Returns
	-------
	float

	"""
	return np.absolute(M).sum()

def prox_l1(C, lam):
	"""  Proximal operator of the l1 penalty: 

         argmin ||M||_1 + (1 / 2*lam) ||M - C||^2_F
            M

	"""
	return clip_positive(C - lam) - clip_positive(-C - lam)

def l2(M):
	""" Computes the (squared) l2 penalty:

	   || M ||^2_F

	Parameters
	----------
	M : ndarray, shape (ncomponents, nvoxels)

	Returns
	-------
	float

	"""
	return np.linalg.norm(M, 'fro')**2

def l2_gradient(M):
	""" Computes the gradient of the l2 penalty:
	   || M ||^2_F

	Parameters
	----------
	M : ndarray, shape (ncomponents, nvoxels)

	Returns
	-------
	ndarray, shape (ncomponents, nvoxels)

	"""
	return 2*M

def smoothness(M, L):
	""" Computes the smoothness penalty:

    \sum_k m_k^T L m_k 

    where L is the graph laplacian of the voxel mask

    Parameters
    ----------

    M : ndarray, shape (ncomponents, nvoxels)
    L : ndarray, shape (nvoxels, nvoxels)

    Returns
    -------
    float

	"""
	ML = (L.T.dot(M.T)).T
	return (M * ML).sum()

def smoothness_gradient(M, L):
	""" Computes the gradient of the smoothness penalty

    Parameters
    ----------

    M : ndarray, shape (ncomponents, nvoxels)
    L : ndarray, shape (nvoxels, nvoxels)
    	the graph laplacian of the mask

    Returns
    -------
    ndarray, shape (nvoxels,)

	"""
	ML = (L.T.dot(M.T)).T # dot(M, L)
	return 2 * ML

def squared_loss(Y, W, M):
	""" Computes the squared loss: 

	1/2 ||Y - WM||^2_F

	Parameters
	----------
	Y : ndarray, shape (nframes, nvoxels)
		subject data
	W : ndarray, shape (nframes, ncomponents)
		timecourses
	M : ndarray, shape (ncomponents, nvoxels)
		spatial maps

	Returns
	-------
	float

	"""
	return 0.5 * fro_diff(Y, W.dot(M))

def squared_loss_gradient(WtW, WtY, M):
	""" Computes the gradient wrt M of: ||Y - WM||^2_F 

	Parameters
	----------
	WtW : ndarray, shape (ncomponents, ncomponents)
		timecourses'* timecourses
	WtY : ndarray, shape (ncomponents, nvoxels)
		timecourses'* subject_data
	M : ndarray, shape (ncomponents, nvoxels)
		spatial maps

	Returns
	-------
	ndarray, shape (ncomponents, nvoxels)

	"""
	return (WtW.dot(M) - WtY)


def prox_squared_loss(WtW, WtY, C, lam):
	""" Computes the proximal operator of the squared loss:

		argmin  (1/2) || Y - WM ||^2_F +  (1 / 2*lam) ||M - C||^2_F
		  M

	Parameters
	--------
	Y : ndarray, shape (nframes, nvoxels)
		subject data 
	W : ndarray, shape (nframes, ncomponents)
		timecourses
	C : ndarray, shape (ncomponents, nvoxels)
	lam : float

	Returns
	-------
	float

	"""
	ncomponents = WtW.shape[1]

	A = WtW + (1. / lam) * np.eye(ncomponents)
	B = WtY +  (1. / lam) * C
	return np.linalg.solve(A, B)


def project_spectral_ball(C, k):
	"""
		Projects a matrix onto the spectral norm k-ball.

		argmin ||M - C||^2_F such that ||M||_2 <= k
		  M

	Parameters
	----------
	C : ndarray, shape (ncomponents, nvoxels)
	k : float

	Returns
	-------
	ndarray, shape (ncomponents, nvoxels)

	"""
	U, svals, Vt = scipy.linalg.svd(C, full_matrices=False)
	svals[svals > k] = k
	svals[svals < -k] = -k
	return U.dot(np.diag(svals)).dot(Vt)


def clip_positive(X):
	""" Clips all elements in a matrix to be non-negative

	Parameters
	----------
	X : ndarray, shape (m, n) 

	Returns
	-------
	ndarray, shape (m, n)

	"""
	return np.clip(X, 0, float('inf'))

def pct_sparse(M):
	""" Computes the percentage of elements in a matrix that are near zero 

	Parameters
	----------
	M : ndarray, shape (ncomponents, nvoxels)

	Returns
	-------
	float

	"""
	return (np.abs(M) < 1e-3).sum() / float(M.size)

def spectral_lasso_pgm(Y, W, alpha, spectral=1.0, niter=500, tol=1e-10):
	""" Solve the (spectral + l1) problem using the proximal gradient method.

		argmin  || Y - WM ||^2_F + alpha ||M||_1  subject to ||M||_2 <= spectral
		  M

	Parameters
	--------
	Y : ndarray, shape (nframes, nvoxels)
		subject data
	W : ndarray, shape (nframes, ncomponents)
		timecourses
	alpha : float
		strength of l1 penalty (higher => more sparse)
	spectral : float
		spectral norm constraint (lower => more distinct)

	Returns
	------
	ndarray, shape (ncomponents, nvoxels)
		spatial maps


	References
	----------
	paper which introduces "generalized forward backward splitting" (proximal gradient for >1 penalties)
		http://icml.cc/2012/papers/674.pdf

	Notes
	----
	we use constant step size (2/lipschitz constant); this works better 
	than a sequence that decreases with sqrt(t)

	NOTE (5/10/2016): I recommend using ADMM rather than PGM to solve spectral lasso

	"""
	def compute_objective(M):
		return squared_loss(Y, W, M) + alpha*l1(M) 

	nframes, nvoxels = Y.shape
	ncomponents = W.shape[1]

	WtW = W.T.dot(W)
	WtY = W.T.dot(Y)

	# lipschitz constant of the gradient of W
	L = np.linalg.norm(WtW, 2) * sqrt(nvoxels)
	L = 2*L # TODO TODO idk why i have to do this; did i get the lipschitz condition wrong?
	eta = 2.0 / L

	M = np.zeros((ncomponents, nvoxels))
	Z1 = np.zeros((ncomponents, nvoxels))
	Z2 = np.zeros((ncomponents, nvoxels))

	objective = float('inf')

	for t in range(niter):
		gradient = squared_loss_gradient(WtW, WtY, M)
		# print L, np.linalg.norm(gradient, 'fro') / np.linalg.norm(M, 'fro')
		Z1 = Z1 + (prox_l1(2*M - Z1 - eta*gradient, 2*eta*alpha) - M)
		Z2 = Z2 + (project_spectral_ball(2*M - Z2 - eta*gradient, 1.0) - M)
		M = 0.5*(Z1 + Z2)

		if t % 10 == 0:
			last_objective = objective
			objective = compute_objective(M)
			print t, objective, np.linalg.norm(M, 2), last_objective - objective, pct_sparse(M)

			if t > 0 and (last_objective - objective) / last_objective <= tol:
				break

	print compute_objective(M), np.linalg.norm(M, 2), pct_sparse(M)

	return M

def spectral_lasso_admm(Y, W, alpha, spectral=1.0, rho=1000.0, niter=200, overrelax=1.0, eps_abs=1e-4, eps_rel=1e-3):
	"""  Solve the spectral lasso problem using the alternating direction method of multipliers

		argmin  || Y - WM ||^2_F + alpha ||M||_1  subject to ||M||_2 <= spectral
		  M

	Parameters
	--------
	Y : ndarray, shape (nframes, nvoxels)
		subject data 
	W : ndarray, shape (nframes, ncomponents)
		timecourses
	alpha : float
		strength of l1 penalty (higher => more sparse)
	spectral : float
		spectral norm constraint (lower => more distinct)
	rho : float
		controls how much the algorithm focuses on the spectral constraint vs. the l1 penalty
	overrelax : float in [1.0, 2.0]
		overrelaxation parameter -- interpolate between the last iterate and the current one
    eps_abs : float
    	absolute tolerance
    eps_rel : float
    	relative tolerance

    Returns
    -------
    ndarray, shape (ncomponents, nvoxels)
    	spatial maps

    References
    ----------
    Stephen Boyd, "Distributed Optimization via the Alternating Direction Method of Multipliers"
    https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

    in particular, pages 18-20 of this manuscript explain the stopping conditions and overrelaxation

    NOTE (5/10/2016): overrelaxation does not seem to help, therefore it is turned off (overrelax=1.0) by default


   	Notes
   	-----
   	This algorithm is VERY sensitive to the tuning parameter rho, and you should tune it for your own dataset.
   	rho should be tuned whenever you change alpha significantly


   	here is typical output:

   	0  15886041.5974 0.068088856749 0.616100201125 1.96650919068 1.72156770924 1205.20095694
	10 15887553.5269 0.348822625477 0.476168900138 1.02054621148 0.0941786149665 59.0385238706
	20 15887544.8867 0.491158391901 0.49119228944 1.00200432807 0.00955570689104 6.75669561474
	28 iters completed

	here's what the columns ^ mean:
	iter | objective | sparsity in M | sparsity in Z1 | spectral norm of M | norm of primal residual | norm of dual residual

	if the optimization ran to completion, the third and fourth colums should be basically identical, and the fifth column
	should be very close to 1

	if the optimization algorithm is taking longer than 150 iterations, you probably need to tune rho


	"""
	def compute_objective(M, Z1):
		return squared_loss(Y, W, M) + alpha*l1(Z1) 

	nframes, nvoxels = Y.shape
	ncomponents = W.shape[1]

	M = np.zeros((ncomponents, nvoxels))

	Z1 = np.zeros((ncomponents, nvoxels))
	Z2 = np.zeros((ncomponents, nvoxels))
	U1 = np.zeros((ncomponents, nvoxels))
	U2 = np.zeros((ncomponents, nvoxels))

	WtW = W.T.dot(W)
	WtY = W.T.dot(Y)

	n = ncomponents * nvoxels
	m = ncomponents * nvoxels
	p = ncomponents * nvoxels * 2

	for iter in range(niter):
		M = prox_squared_loss(WtW, WtY, 0.5 * (Z1 + Z2 - U1 - U2), 0.5 / rho)

		M1_relax = overrelax * M + (1 - overrelax) * Z1
		M2_relax = overrelax * M + (1 - overrelax) * Z2

		Z1_next = prox_l1(U1 + M1_relax, alpha / rho)
		Z2_next = project_spectral_ball(U2 + M2_relax, spectral)

		res_primal_norm = fro_stack(M - Z1_next, M - Z2_next)
		res_dual_norm = rho*fro_stack(Z1 - Z1_next, Z2 - Z2_next)

		eps_primal = sqrt(p) * eps_abs + eps_rel * max(fro_stack(M, M), fro_stack(Z1, Z2))
		eps_dual = sqrt(n) * eps_abs + eps_rel * fro_stack(rho*U1, rho*U2)

		if res_primal_norm < eps_primal and res_dual_norm < eps_dual:
			break

		U1 = U1 + M1_relax - Z1_next
		U2 = U2 + M2_relax - Z2_next

		Z1 = Z1_next
		Z2 = Z2_next

		if iter % 10 == 0:
			print iter, compute_objective(M, Z1), pct_sparse(M), pct_sparse(Z1), np.linalg.norm(M, 2), res_primal_norm, res_dual_norm

	print "%d iters completed" % iter
	print compute_objective(M, M), np.linalg.norm(M, 2), pct_sparse(M)

	return np.asarray(M)

def fro_stack(X, Y):
	""" Stacks two matrices on top of each other and returns the frobenius norm """
	return sqrt(np.linalg.norm(X, 'fro') ** 2 + np.linalg.norm(Y, 'fro')**2)

def spectral_lasso_loss(Y, W, M, alpha):
	""" Loss function for (l1 + spectral) regression

	Parameters
	---------
	Y : ndarray, shape (nframes, nvoxels)
	 	subject cata
	W : ndarray, shape (nframes, ncomponents)
	 	timecourses
	alpha : float
		strength of l1 penalty

	Returns
	-------
	M : ndarray, shape (ncomponents, nvoxels)
		spatial maps

	"""
	return squared_loss(Y, W, M) + alpha*l1(M)
	
def regress_spectral_lasso(Y, W, alpha, spectral=1.0, algorithm='admm', rho=1000.0):
	"""  Solve the spectral lasso problem using one of two algorithms: ADMM or PGM

		argmin  || Y - WM ||^2_F + alpha ||M||_1  subject to ||M||_2 <= spectral
		  M

	Parameters
	--------
	Y : ndarray, shape (nframes, nvoxels)
		subject data 
	W : ndarray, shape (nframes, ncomponents)
		timecourses
	alpha : float
		strength of l1 penalty (higher => more sparse)
	spectral : float
		spectral norm constraint (lower => more distinct)
	algorithm : 'admm' | 'pgm'
		whether to use ADMM (alternating direction method of multipliers) or PGM (proximal gradient method)
	rho : float
		ADMM uses this tuning parameter

	Returns
	-------
	M : ndarray, shape (ncomponents, nvoxels)
		spatial maps

	"""

	if algorithm == 'admm':
		# return spectral_lasso_admm(Y, W, alpha, spectral, rho=1000.0, niter=500) # for spectral <= 100
		# return spectral_lasso_admm(Y, W, alpha, spectral, rho=100.0, niter=500) # for spectral = 100 - 2000
		return spectral_lasso_admm(Y, W, alpha, spectral, rho=rho, niter=500)  # for spectral = 2000 - 3000
	else:
		return spectral_lasso_pgm(Y, W, alpha, spectral)

def compute_laplacian(mask):
	""" Computes the graph laplacian of a voxel mask 


	Parameters
	----------
	mask : ndarray, type boolean, shape (nx, ny, nz)

	Returns
	-------
	sparse matrix (CSR format), shape (nvoxels, nvoxels)
		the graph laplacian

	"""
	nvoxels = mask.sum()
	degree = sparse.lil_matrix((nvoxels, nvoxels))
	adjacency = sparse.lil_matrix((nvoxels, nvoxels))

	coords_to_mask_index = {}
	v = 0 
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			for k in range(mask.shape[2]):
				if mask[i,j,k]:
					coords_to_mask_index[(i,j,k)] = v
					v += 1


	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			for k in range(mask.shape[2]):
				deg = 0
				if mask[i,j,k]:
					if mask[i+1,j,k]:
						deg += 1
						adjacency[coords_to_mask_index[(i,j,k)], coords_to_mask_index[(i+1,j,k)]] = 1
					if mask[i-1,j,k]:
						deg += 1
						adjacency[coords_to_mask_index[(i,j,k)], coords_to_mask_index[(i-1,j,k)]] = 1
					if mask[i,j+1,k]:
						deg += 1
						adjacency[coords_to_mask_index[(i,j,k)], coords_to_mask_index[(i,j+1,k)]] = 1
					if mask[i,j-1,k]:
						deg += 1
						adjacency[coords_to_mask_index[(i,j,k)], coords_to_mask_index[(i,j-1,k)]] = 1
					if mask[i,j,k+1]:
						deg += 1
						adjacency[coords_to_mask_index[(i,j,k)], coords_to_mask_index[(i,j,k+1)]] = 1
					if mask[i,j,k-1]:
						deg += 1
						adjacency[coords_to_mask_index[(i,j,k)], coords_to_mask_index[(i,j,k-1)]] = 1

					degree[coords_to_mask_index[(i,j,k)], coords_to_mask_index[(i,j,k)]] = deg

	laplacian = degree.tocsr() - adjacency.tocsr()

	return laplacian

def wedge_loss(Y, W, M, alpha, theta):
	""" Loss function for wedge regression

	Parameters
	----------
	Y : ndarray, shape (nframes, nvoxels)
		subject data 
	W : ndarray, shape (nframes, ncomponents)
		timecourses
	alpha : float
		overall weight of wedge penalty
	theta : float
		stength of l2 penalty on spatial maps
		the wedge penalty is convex if: theta >= (ncomponents - 1)

	Returns
	-------
	float 

	"""
	ncomponents = W.shape[1]
	def wedge(M):
		""" Compute the wedge penalty """
		kernel = np.ones((ncomponents, ncomponents))
		for k in range(ncomponents):
			kernel[k,k] = theta

		return np.abs(kernel * M.dot(M.T)).sum()

	return squared_loss(Y, W, M) + alpha*wedge(M)

def regress_wedge(Y, W, alpha, theta):
	""" Wedge regression: l1-penalize the inner products between the spatial maps.

	omega(M) =  \sum_{k1 != k2} |m_{k1}^T m_{k2}| + theta \sum_k ||m_k||^2

	Parameters
	--------
	Y : ndarray, shape (nframes, nvoxels)
		subject data 
	W : ndarray, shape (nframes, ncomponents)
		timecourses
	alpha : float
		overall weight of wedge penalty
	theta : float
		stength of l2 penalty on spatial maps
		the wedge penalty is convex if: theta >= (ncomponents - 1)

	Returns
	-------
	M : ndarray, shape (ncomponents, nvoxels)

	"""
	nframes, nvoxels = Y.shape
	ncomponents = W.shape[1]

	WtW = W.T.dot(W)
	WtY = W.T.dot(Y)

	# kernel: off-diagonal elements are 1, on-diagonal elements are theta
	kernel = np.ones((ncomponents, ncomponents))
	for k in range(ncomponents):
		kernel[k,k] = theta

	def wedge(M):
		""" Compute the wedge penalty """
		return np.abs(kernel * M.dot(M.T)).sum()

	def wedge_subgradient(M):
		""" Compute the subgradient of the wedge penalty """
		subgradient = np.zeros((ncomponents, nvoxels))

		MMt = M.dot(M.T)
		MMt_K = np.sign(MMt) * kernel

		for k in range(ncomponents):
			for j in range(ncomponents):
				x = MMt_K[k,j] * M[j,:]
				subgradient[k,:] += np.squeeze(x.T)

		return np.asarray(subgradient)

	def compute_objective(M):
		""" Compute the objective function for wedge regression """
		return squared_loss(Y, W, M) + alpha*wedge(M)

	def compute_objective_subgradient(M):
		""" Compute a subgradient of the objective function """
		return np.asarray(squared_loss_gradient(WtW, WtY, M)) + alpha*wedge_subgradient(M)

	def subgradient_descent(init_step_size, max_iter=2000):
		""" Subgradient descent with step sizes that decrease with 1/sqrt(t) """

		M = np.zeros((ncomponents, nvoxels))
		objective = float('inf')

		for iter in range(1, max_iter + 1):
			subgradient = compute_objective_subgradient(M)
			step_size = init_step_size / sqrt(float(iter))
			M = M - step_size * subgradient

			if iter % 100 == 0:
				last_objective = objective
				objective = compute_objective(M)
				print "%d\t%f\t%f" % (iter, objective, last_objective - objective)
				if objective > last_objective:
					break

		did_finish = (iter == max_iter)
		return M, iter, did_finish

	best_objective = float('inf')
	best_M = None
	for init_step_size in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
		M, niter, did_finish = subgradient_descent(init_step_size)
		if did_finish: # if subgradient descent completed all iterations the step size was too small, so we can just stop
			return best_M

		objective = compute_objective(M)
		if objective < best_objective:
			best_objective = objective
			best_M = M

	return best_M


def fro_diff(X, Y):
	""" Efficiently computes ||X - Y||_F^2 without forming a new matrix
	in memory.

	Parameters
	----------
	X : ndarray, shape (M, N)
	Y : ndarray, shape (M, N)

	Returns
	-------
	float

	"""
	diff = 0
	M, N = X.shape
	for i in range(M):
		diff += np.linalg.norm(X[i,:] - Y[i,:], 2) ** 2
	return diff

def project_stiefel_set(C):
	"""
		Projects a matrix onto the stiefel manifold.

		argmin ||W - C||^2_F such that W'W = I
		  W

	Parameters
	----------
	C : ndarray, shape (nvoxels, ncomponents)

	Returns
	-------
	ndarray, shape (nvoxels, ncomponents)

	"""
	U, S, Vt = np.linalg.svd(C, full_matrices=False)
	return U.dot(Vt)

def prox_squared_loss_stiefel(YtW, C, rho):
	""" argmin_M || Y - W M||^2_F + (rho / 2) || M - C ||^2_F subject to M M' = c^2 I """ 
	U, S, Vt = np.linalg.svd(YtW + (rho) * C.T, full_matrices=False)
	return U.dot(Vt).T

def regress_ortho_sparse1(Y, W, alpha, rho=10.0, niter=200, outfile=None):
	"""  Solve the spectral lasso problem using the alternating direction method of multipliers

		argmin  || Y - WM ||^2_F + alpha ||M||_1  subject to ||M||_2 <= spectral
		  M

	Parameters
	--------
	Y : ndarray, shape (nframes, nvoxels)
		subject data 
	W : ndarray, shape (nframes, ncomponents)
		timecourses
	alpha : float
		strength of l1 penalty (higher => more sparse)
	spectral : float
		spectral norm constraint (lower => more distinct)
	rho : float
		controls how much the algorithm focuses on the spectral constraint vs. the l1 penalty

    Returns
    -------
    ndarray, shape (ncomponents, nvoxels)
    	spatial maps

    References
    ----------
    Stephen Boyd, "Distributed Optimization via the Alternating Direction Method of Multipliers"
    https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

    in particular, pages 18-20 of this manuscript explain the stopping conditions and overrelaxation


	"""
	def compute_objective(M):
		return squared_loss(Y, W, M) + alpha*l1(M) 

	nframes, nvoxels = Y.shape
	ncomponents = W.shape[1]

	M = np.zeros((ncomponents, nvoxels))

	Z1 = np.zeros((ncomponents, nvoxels))
	Z2 = np.zeros((ncomponents, nvoxels))
	U1 = np.zeros((ncomponents, nvoxels))
	U2 = np.zeros((ncomponents, nvoxels))

	WtW = W.T.dot(W)
	WtY = W.T.dot(Y)

	for iter in range(niter):
		M = prox_squared_loss(WtW, WtY, 0.5 * (Z1 + Z2 - U1 - U2), 0.5 / rho)

		Z1_next = prox_l1(U1 + M, alpha / rho)
		Z2_next = project_stiefel_set(U2 + M)

		res_primal_norm = fro_stack(M - Z1_next, M - Z2_next)
		res_dual_norm = rho*fro_stack(Z1 - Z1_next, Z2 - Z2_next)

		U1 = U1 + M - Z1_next
		U2 = U2 + M - Z2_next

		Z1 = Z1_next
		Z2 = Z2_next

		if iter % 10 == 0:
			print iter, compute_objective(M), pct_sparse(M), pct_sparse(Z1), res_primal_norm, res_dual_norm

	print "%d iters completed" % iter
	print compute_objective(M), pct_sparse(M)

	return np.asarray(M)


def regress_ortho_sparse2(Y, W, alpha, rho=10.0, niter=200, outfile=None):
	""" MADMM is equivalent to ADMM with the splitting 

	f(X) = 0.5 *||Y - W X ||^2_F + I_{XX' = I} 
	g(Z) = alpha * ||Z||_1

	"""

	def compute_objective(X, Z):
		return squared_loss(Y, W, X) + alpha*l1(Z) 

	nframes, nvoxels = Y.shape
	ncomponents = W.shape[1]

	X = np.zeros((ncomponents, nvoxels))
	Z = np.zeros((ncomponents, nvoxels))
	U = np.zeros((ncomponents, nvoxels))

	YtW = Y.T.dot(W)

	if outfile:
		f = open(outfile, 'w')

	for iter in range(niter):
		X = prox_squared_loss_stiefel(YtW, (Z - U), rho)
		Z = prox_l1(U + X, alpha / rho)

		U = U + X - Z

		if iter % 10 == 0:
			string = '%d\t%f\t%s\t%s\n' % (iter, compute_objective(X, Z), pct_sparse(X), pct_sparse(Z))
			if outfile:
				f.write(string)
			else:
				print string

	if outfile:
		f.close()

	return np.asarray(X)

