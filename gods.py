# Copyright (c) 2019-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

""" Generalized One-Class Discriminative Subspaces (GODS) for anomaly detection.
    Implementation in Python using PyManOpt.
"""
import argparse
import os
import pdb
import pickle
import random

import autograd.numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Euclidean, Oblique, Product, Sphere, Stiefel
from pymanopt.solvers import ConjugateGradient, TrustRegions
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import normalize


def init_subspaces(X, num_subspaces, eta):
    if num_subspaces > 1:
        w1, w2, b1, b2 = init_stiefel(X, num_subspaces, eta)
        # w1,_,_ = np.linalg.svd(np.transpose(X[idx[:num_subspaces], :]), full_matrices=False)
        # w2,_,_ = np.linalg.svd(np.transpose(X[idx[-num_subspaces:], :]), full_matrices=False)
    else:
        dist = np.sqrt(np.square(X).sum(1))
        idx = np.argsort(dist)
        w1 = X[idx[0], :]
        w2 = X[idx[-1], :]
        w1 /= np.linalg.norm(w1) + 1e-10
        w2 /= np.linalg.norm(w2) + 1e-10

    return (w1, w2, np.zeros((1, num_subspaces), dtype="float"), np.zeros((1, num_subspaces), dtype="float"))


def init_stiefel(X, num_subspaces, eta):
    manifold = Product((Stiefel(d, k), Euclidean(1, k)))
    data = np.transpose(X)
    #    @pymanopt.function.Autograd
    def cost_lower(M):
        w, b = np.transpose(M[0]), np.transpose(M[1])  # the subspaces.
        ww = np.dot(w, data) + b * np.ones((X.shape[0],))
        upper = np.maximum(0, np.add(eta, np.max(ww, axis=0)))
        obj = np.sum(np.square(upper)) + np.sum(np.square(b))
        return obj

    # @pymanopt.function.Autograd
    def cost_upper(M):
        w, b = np.transpose(M[0]), np.transpose(M[1])  # the subspaces.
        ww = np.dot(w, data) + b * np.ones((X.shape[0],))
        lower = np.maximum(0, np.add(eta, -np.min(ww, axis=0)))
        obj = np.sum(np.square(lower)) + np.sum(np.square(b))
        return obj

    solver = ConjugateGradient(maxiter=100)
    problem_upper = Problem(manifold=manifold, cost=cost_upper, verbosity=int(args.verbose) * 3)
    w2, b2 = solver.solve(problem_upper)
    problem_lower = Problem(manifold=manifold, cost=cost_lower, verbosity=int(args.verbose) * 3)
    w1, b1 = solver.solve(problem_lower)
    return w1, w2, b1, b2


seed = 123
print("seed=%d" % (seed))
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GODS: pose anomaly detection.")
    parser.add_argument(
        "--split_num",
        default=0,
        type=int,
        help="in case of cross-validation, specfiy the cross-validation number (1/2/3/4)",
    )
    parser.add_argument(
        "--num_subspaces", default=2, type=int, help="number of hyperplanes in each of GODS subspaces. Default=2"
    )
    parser.add_argument(
        "--eta",
        default=0.01,
        type=float,
        help="eta, controlling how far are the predictions from the hyperplanes. Default=0.01",
    )
    parser.add_argument(
        "--L",
        default=0.001,
        type=float,
        help="lambda, regularization cost for the distance between subspaces. Default=0.001",
    )
    parser.add_argument(
        "--max_iter", default=100, type=int, help="max-number of Riemannian optimization iterations. Default=100"
    )
    parser.add_argument(
        "--optim",
        default="cg",
        type=str,
        help="Optimizer to use. conjugate-gradient (cg) / trust-regions (tr). Default=cg",
    )
    parser.add_argument(
        "--thresh",
        default=0.0,
        type=float,
        help="threshold for deciding if the given data point is within the two subspaces. Default=0",
    )
    parser.add_argument(
        "--unnormalize",
        action="store_true",
        help="Should you normalize all pose features to unit norm before learning? Sometimes helps.",
    )
    parser.add_argument(
        "--embed_path",
        default="./data/poses/",
        type=str,
        help="path to pkl filename to store the embedded pose data. It will be saved as embed_path/data_train<split_num>.pkl and embed_path/data_test<split_num>.pkl",
    )
    parser.add_argument("--verbose", action="store_true", help="echo some messages regarding status of the program.")
    args = parser.parse_args()

    # Data prepare
    # pdb.set_trace()
    with open(os.path.join(args.embed_path, "data_train" + str(args.split_num) + ".pkl"), "rb") as handle:
        data_tr = pickle.load(handle, encoding="latin1")
    with open(os.path.join(args.embed_path, "data_test" + str(args.split_num) + ".pkl"), "rb") as handle:
        data_va_n = pickle.load(handle, encoding="latin1")

    # select part of the training set as normal data for testing. We will use all of the negative data for testing.
    random.seed(seed)
    index = random.sample(range(0, data_tr.shape[0]), data_va_n.shape[0] - 2)
    data_va_p = np.array([data_tr[i, :] for i in range(data_tr.shape[0]) if i in index])
    data_tr = np.array([data_tr[i, :] for i in range(data_tr.shape[0]) if i not in index])
    data_va = np.concatenate((data_va_n, data_va_p), axis=0)
    label_va = np.ones((data_va.shape[0],), dtype=int)
    label_va[: data_va.shape[0] // 2] = 0
    label_tr = np.ones((data_tr.shape[0],), dtype=int)

    # data normalization. Make the embedded features unit norm.
    if not args.unnormalize:
        np.random.seed(seed)
        data_tr = normalize(data_tr, axis=1, norm="l2")
        data_va = normalize(data_va, axis=1, norm="l2")

    # Manifold setting
    d = data_tr.shape[1]  # data dimensionality
    k = args.num_subspaces  # number of subspaces.
    eta = args.eta  # 0.01
    Lambda = args.L  # 0.001
    num_pts = len(data_tr)  # number of points.

    # define the objective for optimization.
    # @pymanopt.function.Autograd
    def cost(M):
        w1, w2, b1, b2 = (
            np.transpose(M[0]),
            np.transpose(M[1]),
            np.transpose(M[2]),
            np.transpose(M[3]),
        )  # the subspaces.
        data = np.transpose(data_tr)

        ww1 = np.dot(w1, data) + b1 * np.ones((data_tr.shape[0],))
        ww2 = np.dot(w2, data) + b2 * np.ones((data_tr.shape[0],))

        lower = np.maximum(0, np.add(eta, -np.min(ww1, axis=0)))
        upper = np.maximum(0, np.add(eta, np.max(ww2, axis=0)))

        obj = (
            np.sum(np.square(lower))
            + np.sum(np.square(upper))
            + Lambda * (np.sum(np.square(ww1)) + np.sum(np.square(ww2)))
        )

        return obj

    # compute prediction accuracy.
    def calculate_accruacy(opt, data, label_gt):
        label = []
        w1, w2, b1, b2 = np.transpose(opt[0]), np.transpose(opt[1]), np.transpose(opt[2]), np.transpose(opt[3])
        for i in range(len(data)):
            item = np.transpose(data[i, :])
            ww1 = np.expand_dims(np.matmul(w1, item), axis=1) + b1
            ww2 = np.expand_dims(np.matmul(w2, item), axis=1) + b2

            # we use exact thresholds for judging accuracy. That is, if the pose falls between the two subspaces, we take that as normal class, else abnormal.
            if np.min(ww1) > args.thresh and np.max(ww2) < args.thresh:
                label.append(1)
            else:
                label.append(0)
        accuracy = accuracy_score(label_gt, label)
        F1 = f1_score(label_gt, label)
        precision, recall, fbeta_score, support = precision_recall_fscore_support(label_gt, label)
        return accuracy, F1, precision, recall

    # setup the manopt framework.
    if k > 1:
        manifold = Product(
            (Stiefel(d, k), Stiefel(d, k), Euclidean(1, k), Euclidean(1, k))
        )  # product manifold of two Stiefels and their biases.
    else:
        # if we use only one subspace, then its better to use Sphere manifold for efficiency.
        manifold = Product(
            (Sphere(d), Sphere(d), Euclidean(1, k), Euclidean(1, k))
        )  # product manifold of two Stiefels and their biases.

    problem = Problem(manifold=manifold, cost=cost, verbosity=int(args.verbose) * 3)  # problem setup.
    if args.optim == "cg":
        solver = ConjugateGradient(
            maxiter=args.max_iter
        )  # we use Riemannian Conjugate gradient. Another option is to use TrustRegions.
    elif args.optim == "tr":
        solver = ConjugateGradient(
            maxiter=args.max_iter
        )  # we use Riemannian Conjugate gradient. Another option is to use TrustRegions.
    else:
        print("unknown solver. options are --optim='cg' or --optim='tr'")

    init_X = init_subspaces(data_tr, k, eta)
    Xopt = solver.solve(
        problem, x=init_X
    )  # solve the problem. This will use autograd for computing the gradients automatically.

    # compute the anomaly detection peformance.
    accu_tr, _, _, _ = calculate_accruacy(Xopt, data_tr, label_tr)
    accu_va, F1, precision, recall = calculate_accruacy(Xopt, data_va, label_va)
    print("split status")
    print("-------------------")
    print(
        "num train = %d \n num test = %d \n num test normal = num test abnormal = %d\n"
        % (data_tr.shape[0], data_va.shape[0], data_va_n.shape[0] - 1)
    )
    print("-------------------")
    print("Training accuracy is %.2f" % (accu_tr))
    print("Test Evaluation:")
    print(
        "Testing accuracy is %.2f" % (accu_va),
        "\nF1 score is %.2f " % (F1),
        "\n Precision (abnormal samples) = %.2f " % (precision[0]),
        "\n Recall (abnormal samples) =  %.2f" % (recall[0]),
    )
