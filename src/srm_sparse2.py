#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

""" orthogonal AND sparse SRM """

import numpy as np
import argparse
import os
from regress import regress_ortho_sparse1, regress_ortho_sparse2, squared_loss, l1
from srm_am import srm, srm_local, add_srm_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse SRM')
    parser.add_argument('algorithm', choices=['1', '2'])
    parser.add_argument('alpha', type=float, default=2.0, help='strength of l1 penalty')
    parser.add_argument('--rho', type=float, default=1000.0)
    parser.add_argument('--local', action='store_true')
    add_srm_args(parser)
    args = parser.parse_args()

    identifier = 'srm_sparse_k_%d_alpha_%s_%s_%s' % (args.ncomponents, args.alpha, args.split, args.half)

    output_directory = os.path.join(args.directory, identifier)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    if args.algorithm == '1':
        # honestly niter=200 is probably not enough
        learn_subject_maps = lambda subject_data, timecourses, outfile : regress_ortho_sparse1(subject_data, timecourses, args.alpha, rho=args.rho, niter=200, outfile=outfile)
    elif args.algorithm == '2':
        learn_subject_maps = lambda subject_data, timecourses, outfile : regress_ortho_sparse2(subject_data, timecourses, args.alpha, rho=args.rho, niter=100, outfile=outfile)
    compute_objective = lambda subject_data, timecourses, maps : squared_loss(subject_data, timecourses, maps) + args.alpha*l1(maps)
    if args.local:
        srm_local(args.ncomponents, output_directory, args.dataset, args.split, args.half, learn_subject_maps, are_orthogonal=True, compute_objective=compute_objective, initial_iter=args.initial_iter, niter=args.niter)
    else:
        srm(args.ncomponents, output_directory, args.dataset, args.split, args.half, learn_subject_maps, are_orthogonal=True, compute_objective=compute_objective, initial_iter=args.initial_iter, niter=args.niter)
