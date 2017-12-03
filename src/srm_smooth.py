#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

import numpy as np
import argparse
import os
from regress import regress_smooth_lasso, compute_laplacian, smooth_lasso_loss
from srm_am import srm, add_srm_args

def get_laplacian(mask_path):
    mask_data = np.load(mask_path)
    laplacian = compute_laplacian(mask_data)
    return laplacian

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smooth SRM')
    parser.add_argument('mask_path', type=str, help='path to .npy mask')
    parser.add_argument('alpha', type=float, default=2.0, help='strength of l1 penalty')
    parser.add_argument('beta', type=float, default=0.0, help='strength of l2 penalty')
    parser.add_argument('gamma', type=float, default=1.0, help='strength of smoothness penalty')
    add_srm_args(parser)
    args = parser.parse_args()

    laplacian = get_laplacian(args.mask_path)

    identifier = 'srm_smooth_k_%d_alpha_%s_beta_%s_gamma_%s_%s_%s' % (args.ncomponents, args.alpha, args.beta, args.gamma, args.split, args.half)

    output_directory = os.path.join(args.directory, identifier)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    learn_subject_maps = lambda subject_data, timecourses : regress_smooth_lasso(subject_data, timecourses, args.alpha, args.beta, args.gamma, laplacian)
    compute_objective = lambda subject_data, timecourses, maps : smooth_lasso_loss(subject_data, timecourses, maps, args.alpha, args.beta, args.gamma, laplacian)
    srm(args.ncomponents, output_directory, args.dataset, args.split, args.half, learn_subject_maps, compute_objective=compute_objective, initial_iter=args.initial_iter, niter=args.niter)

