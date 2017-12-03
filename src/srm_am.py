""" Base class for alternating minimization for SRM. """

import numpy as np
import os
import cvxpy as cvx
from datasets import load_dataset
from multiprocessing import Process, Queue

def percent_sparse(M):
    """ Computes the percentage of elements in a matrix that are near zero 

    Parameters
    ----------
    M : ndarray, shape (ncomponents, nvoxels)

    Returns
    -------
    float

    """
    return (np.abs(M) < 1e-4).sum() / float(M.size)

def collate_matrices(allsubjects_data, maps):
    """ Computes summary statistics from data across multiple subjects

    Parameters
    ----------
    allsubjects_data : list of ndarray, shape (nframes, nvoxels)
    maps : list of ndarray, shape (ncomponents, nvoxels)

    Returns
    -------
    YMt : ndarray, shape (nframes, ncomponents)
       \sum_{i=1}^{num subjects} Y_i * M_i'
    MMt : ndarray, shape (ncomponents, ncomponents)
       \sum_{i=1}^{num subjects} M_i * M_i'

    """
    nsubjects = len(allsubjects_data)
    nframes, nvoxels = allsubjects_data[0].shape
    ncomponents = maps[0].shape[0]

    YMt = np.zeros((nframes, ncomponents))
    MMt = np.zeros((ncomponents, ncomponents))

    for i in range(nsubjects):
        subject_data = allsubjects_data[i]
        YMt += subject_data.dot(maps[i].T)
        MMt = maps[i].dot(maps[i].T)

    return (YMt, MMt)

def save_timecourses(timecourses, directory, iter):
    """ Saves timecourses to disk (every 10 iterations)

    Parameters
    ----------
    timecourses : ndarray, shape (nframes, ncomponents)
    directory : string
    iter : int
    
    """
    if iter % 10 == 8:
        np.save("%s/iter%d_timecourses.npy" % (directory, iter), timecourses)

def save_maps(maps, i, directory, iter):
    """ Saves a single subject's maps to disk (every 10 iterations)

    Parameters
    ----------
    maps : ndarray, shape (ncomponents, nvoxels)
    i : int
        the subject index
    directory : string
    iter : int
    
    """
    # if iter % 10 == 9:
    np.save("%s/iter%d_subject%d_maps.npy" % (directory, iter, i), maps)

def learn_timecourses_bounded(YMt, MMt, verbose=False):
    """ Learn the timecourses, with each timecourse constrained to have less than unit norm.
        Uses cvxpy with a Mosek solver.  Mosek must be installed on the system.

    Parameters
    ----------
    YMt : ndarray, shape (nframes, ncomponents)
        sum over all subjects i of Y_i * M_i'
    MMt : ndarray, shape (ncomponents, ncomponents)
        sum over all subjects i of M_i * M_i'

    Returns
    -------
    timecourses : ndarray, shape (nframes, ncomponents)

    """
    nframes, ncomponents = YMt.shape

    L = np.linalg.cholesky(MMt)

    W = cvx.Variable(nframes, ncomponents)
    objective = cvx.Minimize(-2*cvx.trace(YMt.T*W) + cvx.sum_squares(W*L))

    # non-negative constraint
    constraints = []

    # norm constraints
    for k in range(ncomponents):
        constraints.append(cvx.norm(W[:,k]) <= 1)

    problem = cvx.Problem(objective, constraints)
    problem.solve(solver='MOSEK', verbose=verbose, max_iters=100000)

    timecourses = W.value
    return np.asarray(timecourses)


def learn_subject_maps_multi(subject_data_path, i, timecourses, directory, iter, queue, learn_subject_maps, compute_objective):
    """ Learns a single subject's spatial maps given the subject data and the timecourses.  This function is called when
    SRM is run in parallel over multiple processors on a single machine. 

    Loads the subject data, learns the subject maps, and saves them.  Computes YMt and MMt and puts them in a queue
    so that they can be used by the caller.

    Parameters
    ----------
    subject_data_path : string
        the path to the subject's data on disk, stored as an ndarray, shape (nframes, nvoxels)
    i : int
        which subject to learn (this is used to to save the spatial maps in the right place)
    timecourses : ndarray, shape (nframes, ncomponents)
    directory : string
        output directory
    iter : int
        current top-level iteration (this is used to save the spatial maps in the right place)
    queue : multiprocessing.Queue
        a queue in which some summary statistics are returned
    learn_subject_maps : function: subject_data, timecourses -> subject maps
        a function which learns subject maps given subject data and timecourses
    compute_objective : function: subject_data, timecourses, maps -> float
        a function which computes the objective function

    Side effects
    ------------
    saves spatial maps on disk
    puts summary statistics in queue

    """
    subject_data = np.load(subject_data_path % i)
    maps = learn_subject_maps(subject_data, timecourses, '%s/admm_iter%d_subject%d.txt' % (directory, iter, i))
    save_maps(maps, i, directory, iter)
    YMt = subject_data.dot(maps.T)
    MMt = maps.dot(maps.T)

    if compute_objective == None:
        queue.put((YMt, MMt))
    else:
        objective = compute_objective(subject_data, timecourses, maps)
        sparsity = percent_sparse(maps)
        queue.put((YMt, MMt, objective, sparsity))

    del subject_data

def load_allsubjects_data(subject_data_path, subject_ids):
    """ Load all subjects' data from disk.

    Parameters
    ----------
    subject_data_path : string
        the path to any arbitrary subject's data, with a %d placeholder for the subject id
    subject_ids : list of int
        the ids of subjects whose data should be loaded

    Returns
    -------
    allsubjects_data : list of ndarray, shape (nframes, nvoxels)

    """
    allsubjects_data = []
    for i in subject_ids:
        allsubjects_data.append(np.load(subject_data_path % i))

    return allsubjects_data

def srm(ncomponents, directory, dataset, split, half, learn_subject_maps, compute_objective=None, initial_iter=0, niter=100, are_orthogonal=False):
    """ Alternating minimization for a shared response model.  
        Parallelizes the learn-maps stage across many cores.

    Parameters
    ----------
    ncomponents : int
        how many components to learn
    directory : string
        the output directory to put everything in
    dataset : Pieman | TZ | Sherlock
        which dataset to learn from
    split : full | left | right
        which half of the movie to learn
    half : first | second
        which subjects to learn from
    learn_subject_maps : function: subject_data, timecourses -> subject maps
        a function which learns subject maps given subject data and timecourses
    compute_objective : function: subject_data, timecourses, maps -> float
        a function which computes the objective function
    are_orthogonal : boolean
        whether the spatial maps are orthogonal.
        if true, learning the timecourses is just an orthogonal projection (no constrained least squares solver needed)


    Side effects
    ------------
    saves spatial maps and timecourses every 10 iterations to disk, along with trace of objective function

    """

    if os.path.exists('%s/objectives.txt' % directory):
        exit()

    subject_data_path, subject_ids = load_dataset(dataset, split, half)
    nsubjects = len(subject_ids)

    allsubjects_data = load_allsubjects_data(subject_data_path, subject_ids)

    nframes = allsubjects_data[0].shape[0]
    nvoxels = [allsubjects_data[i].shape[1] for i in range(nsubjects)]

    maps = [np.zeros((ncomponents, nvoxels[i])) for i in range(nsubjects)]

    for i in range(nsubjects):
        A = np.mat(np.random.random((nvoxels[i],ncomponents)))
        Q, R_qr = np.linalg.qr(A)
        maps[i] = Q.T
    
    # summary statistics 
    (YMt, MMt) = collate_matrices(allsubjects_data, maps)
    del allsubjects_data

    subject_objectives = np.zeros((niter/2, nsubjects))
    subject_sparsities = np.zeros((niter/2, nsubjects))
    objectives = np.zeros(niter/2)

    for iter in range(initial_iter, niter):
        print "iter %d" % iter

        if iter % 2 == 0:
            # learn timecourses
            if are_orthogonal:
                timecourses = YMt / float(nsubjects)
            else:
                timecourses = learn_timecourses_bounded(YMt, MMt)
            save_timecourses(timecourses, directory, iter)
        else:
            # learn spatial maps
            queues = [Queue() for i in range(nsubjects)]
            workers = [Process(target=learn_subject_maps_multi, args=(subject_data_path, subject_ids[i], timecourses, directory, iter, queues[i], learn_subject_maps, compute_objective)) for i in range(nsubjects)]

            for w in workers:
                w.start()
            results = [queues[i].get() for i in range(nsubjects)]
            for w in workers:
                w.join()

            YMt = np.zeros((nframes, ncomponents))
            MMt = np.zeros((ncomponents, ncomponents))
            for i in range(nsubjects):
                YMt += results[i][0]
                MMt += results[i][1]

                if compute_objective != None:
                    subject_objectives[(iter-1)/2, i] = results[i][2]
                    subject_sparsities[(iter-1)/2, i] = results[i][3]

            objectives = subject_objectives.sum(1)

            np.savetxt('%s/objectives.txt' % directory, objectives)
            np.savetxt('%s/subject_objectives.txt' % directory, subject_objectives)
            np.savetxt('%s/subject_sparsities.txt' % directory, subject_sparsities)


def srm_local(ncomponents, directory, dataset, split, half, learn_subject_maps, compute_objective=None, initial_iter=0, niter=100, are_orthogonal=False):
    """ Alternating minimization for a shared response model, not parallelized.

    Parameters
    ----------
    ncomponents : int
        how many components to learn
    directory : string
        the output directory to put everything in
    dataset : Pieman | TZ | Sherlock
        which dataset to learn from
    split : full | left | right
        which half of the movie to learn
    half : first | second
        which subjects to learn from
    learn_subject_maps : function: subject_data, timecourses -> subject maps
        a function which learns subject maps given subject data and timecourses
    compute_objective : function: subject_data, timecourses, maps -> float
        a function which computes the objective function
    are_orthogonal : boolean
        whether the spatial maps are orthogonal.
        if true, learning the timecourses is just an orthogonal projection (no constrained least squares solver needed)

    """
    subject_data_path, subject_ids = load_dataset(dataset, split, half)
    nsubjects = len(subject_ids)

    allsubjects_data = load_allsubjects_data(subject_data_path, subject_ids)

    nframes = allsubjects_data[0].shape[0]
    nvoxels = [allsubjects_data[i].shape[1] for i in range(nsubjects)]

    maps = [np.zeros((ncomponents, nvoxels[i])) for i in range(nsubjects)]

    for i in range(nsubjects):
        A = np.mat(np.random.random((nvoxels[i],ncomponents)))
        Q, R_qr = np.linalg.qr(A)
        maps[i] = Q.T

    objectives = np.zeros(niter/2)

    for iter in range(initial_iter, niter):
        print "iter %d" % iter

        if iter % 2 == 0:
            # learn timecourses
            (YMt, MMt) = collate_matrices(allsubjects_data, maps)
            if are_orthogonal:
                timecourses = YMt / float(nsubjects)
            else:
                timecourses = learn_timecourses_bounded(YMt, MMt)
            save_timecourses(timecourses, directory, iter)
        else:
            objective = 0
            for i in subject_ids:
                maps[i] = learn_subject_maps(allsubjects_data[i], timecourses, None)
                save_maps(maps[i], i, directory, iter)
                objective += compute_objective(allsubjects_data[i], timecourses, maps[i])

            objectives[(iter-1)/2] = objective

        np.savetxt('%s/objectives.txt' % directory, objectives)


def add_srm_args(parser):
    """ Add arguments to an argparse.parser that are common to many SRM models """

    parser.add_argument('directory', metavar='directory', action='store', help='output directory')
    parser.add_argument('ncomponents', metavar='k', type=int, action='store', default=10, help='number of components (default 10)')
    parser.add_argument('--split', action='store', choices=['full', 'left', 'right'], default='full', help='What portion of the data to learn from.')
    parser.add_argument('--half', action='store', choices=['first', 'second'], default='first', help='Which subjects to learn from.')
    parser.add_argument('--dataset', action='store', choices=['Pieman', 'TZ', 'Sherlock', 'Raiders_VT'], default='TZ', help='Which dataset to use.')
    parser.add_argument('--initial_iter', type=int, default=0)
    parser.add_argument('--niter', type=int, default=100)


