#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

import numpy as np
import argparse
import os
from regress import regress_elastic_net, compute_laplacian, elastic_net_loss
from srm_am import srm, add_srm_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse SRM')
    parser.add_argument('alpha', type=float, default=2.0, help='strength of l1 penalty')
    parser.add_argument('beta', type=float, default=0.0, help='strength of l2 penalty')
    add_srm_args(parser)
    args = parser.parse_args()

    identifier = 'srm_smooth_k_%d_alpha_%s_beta_%s_%s_%s' % (args.ncomponents, args.alpha, args.beta, args.split, args.half)

    output_directory = os.path.join(args.directory, identifier)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    learn_subject_maps = lambda subject_data, timecourses : regress_elastic_net(subject_data, timecourses, args.alpha, args.beta)
    compute_objective = lambda subject_data, timecourses, maps : elastic_net_loss(subject_data, timecourses, maps, args.alpha, args.beta)
    srm(args.ncomponents, output_directory, args.dataset, args.split, args.half, learn_subject_maps, compute_objective=compute_objective, initial_iter=args.initial_iter, niter=args.niter)
