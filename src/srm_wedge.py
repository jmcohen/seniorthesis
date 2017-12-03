#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u


import numpy as np
import argparse
import os
from regress import regress_wedge, wedge_loss
from srm_am import srm, srm_local, add_srm_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse SRM')
    parser.add_argument('alpha', type=float, default=1.0, help='overall strength of penalty')
    parser.add_argument('theta', type=float, default=9.0, help='l2 penalty (high = convex, low = nonconvex)')
    add_srm_args(parser)
    args = parser.parse_args()

    identifier = 'srm_wedge_k_%d_alpha_%s_theta_%s_%s_%s' % (args.ncomponents, args.alpha, args.theta, args.split, args.half)

    output_directory = os.path.join(args.directory, identifier)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    learn_subject_maps = lambda subject_data, timecourses : regress_wedge(subject_data, timecourses, args.alpha, args.theta)
    compute_objective = lambda subject_data, timecourses, maps : wedge_loss(subject_data, timecourses, maps, args.alpha, args.theta)
    srm(args.ncomponents, output_directory, args.dataset, args.split, args.half, learn_subject_maps, compute_objective=compute_objective, initial_iter=args.initial_iter, niter=args.niter)

