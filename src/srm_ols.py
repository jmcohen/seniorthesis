#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

import numpy as np
import argparse
import os
from regress import regress_ols, ols_loss
from srm_am import srm, add_srm_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OLS SRM')
    add_srm_args(parser)
    args = parser.parse_args()

    identifier = 'srm_ols_k_%d_%s_%s' % (args.ncomponents, args.split, args.half)

    output_directory = os.path.join(args.directory, identifier)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    learn_subject_maps = lambda subject_data, timecourses : regress_ols(subject_data, timecourses)
    compute_objective = lambda subject_data, timecourses, maps : ols_loss(subject_data, timecourses, maps)
    srm(args.ncomponents, output_directory, args.dataset, args.split, args.half, learn_subject_maps, compute_objective=compute_objective, initial_iter=args.initial_iter, niter=args.niter)
