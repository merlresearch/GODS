# Copyright (c) 2019-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

""" Kernelized One-Class Discriminative Subspaces (KODS) for anomaly detection.
"""
import argparse
import os
import pickle
import random

import autograd.numpy as np
from pymanopt import Problem
from pymanopt.manifolds import Euclidean, GeneralizedStiefel, Oblique, Product, Sphere, Stiefel
from pymanopt.solvers import ConjugateGradient, TrustRegions
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import chi2_kernel

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
    parser.add_argument("--sigma", default=1.0, type=float, help="sigma, std for the rbf kernel, Default=1.0")
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
    parser.add_argument(
        "--kernel",
        default="rbf",
        type=str,
        help="kernel to use for kgods. (currently, its linear/rbf/min/chisq. (default=rbf)",
    )

    args = parser.parse_args()

    # Data prepare
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
        data_tr = data_tr / data_tr.sum(1)[:, np.newaxis]
        data_va = data_va / data_va.sum(1)[:, np.newaxis]

    # Manifold setting
    d = data_tr.shape[1]  # data dimensionality
    k = args.num_subspaces  # number of subspaces.
    eta = args.eta  # 0.01
    sigma = args.sigma
    num_pts = len(data_tr)  # number of points.
    one_kxk = np.ones((k, k), dtype="float")
    one_kxn = np.ones((k, num_pts), dtype="float")
    one_nxk = np.ones((num_pts, k), dtype="float")
    X = data_tr

    def compute_chisq_kernel(A, B, sigma):
        """
        chi-squared kernel
        """
        kernel = chi2_kernel(A, B, gamma=sigma)
        return kernel

    def compute_min_kernel(A, B):
        """
        histogram intersection kernel: min kernel: generalized histogram intersection kernel.
        """
        kernel = np.zeros((A.shape[0], B.shape[0]))
        for k in range(A.shape[1]):
            kernel += np.minimum(A[:, k][:, np.newaxis], B[:, k][:, np.newaxis].transpose())
        kernel = kernel / A.shape[1]
        return kernel

    def compute_kernel(A, B, same=False):
        if args.kernel == "linear":
            kernel = np.matmul(A, B.transpose())  # linear kernel
        elif args.kernel == "rbf":
            kernel = np.exp(-(cdist(A, B, "sqeuclidean")) / (2.0 * sigma))  # RBF kernel
        elif args.kernel == "chisq":
            kernel = compute_chisq_kernel(A, B, sigma)  # chi-sq kernel
        elif args.kernel == "min":
            kernel = compute_min_kernel(A, B)  # histogram intersection kernel.

        if same == True:
            kernel = (
                kernel + kernel.transpose()
            ) / 2.0  # make sure the kernel is symmetric, else complex roots may come up.
            kernel += np.eye(num_pts) * 1e-7  # regulrize the kernel, avoid low-rank for numerical problems.
        print("mean of the kernel=%f" % (kernel.mean()))
        return kernel

    # define the objective for optimization.
    def cost(M):
        Y, Z = M[0], M[1]
        Y = Y * Y
        Z = Z * Z
        obj = (
            0.5 * np.matmul(Y.transpose(), Y).sum()
            + np.trace(np.matmul(Y.transpose(), np.matmul(K, Z)))
            - eta * np.trace(np.matmul((Y - Z).transpose(), one_nxk))
        )

        obj += 0.1 * np.linalg.norm(Y - Z) ** 2.0  # the lagrangian has (Y-Z)*1 = 0. We use a soft variant of that here.
        return obj

    # compute prediction accuracy.
    def calculate_accruacy(opt, data, label_gt):
        label = []
        Y, Z = opt  # this is n x kv
        Y = Y * Y  # this is nxk
        Z = Z * Z

        # estimate b1 and b2.
        b1 = (eta - np.matmul(K, Z)).max(0)
        b2 = (eta + np.matmul(K, Y)).min(0)

        # kernel for test set.
        KK_test = compute_kernel(data, X)

        # classify each test point as positive or negative.
        for i in range(len(data)):
            ww1 = np.matmul(KK_test[i, :][np.newaxis, :], Z) + b1
            ww2 = -np.matmul(KK_test[i, :][np.newaxis, :], Y) + b2

            # we use exact thresholds for judging accuracy. That is, if the pose falls between the two subspaces,
            # we take that as normal class, else abnormal.
            if np.all(ww1 > 0) and np.all(ww2 < 0):
                label.append(1)
            else:
                label.append(0)

        accuracy = accuracy_score(label_gt, label)
        F1 = f1_score(label_gt, label)
        precision, recall, fbeta_score, support = precision_recall_fscore_support(label_gt, label)
        return accuracy, F1, precision, recall

    # compute kernel for training set.
    K = compute_kernel(X, X, same=True)

    # Generalized Stiefel is a new manifold implementation, not available in pymanopt package.
    manifold = Product((GeneralizedStiefel(num_pts, k, K), GeneralizedStiefel(num_pts, k, K)))

    problem = Problem(manifold=manifold, cost=cost, verbosity=3)  # problem setup.
    init_YZ = (np.random.randn(num_pts, k), np.random.randn(num_pts, k))
    if args.optim == "cg":
        solver = ConjugateGradient(maxiter=args.max_iter)
    elif args.optim == "tr":
        solver = ConjugateGradient(maxiter=args.max_iter)
    else:
        print("unknown solver. options are --optim='cg' or --optim='tr'")

    Xopt = solver.solve(
        problem
    )  # , x = init_YZ) # solve the problem. This will use autograd for computing the gradients automatically.
    # compute the anomaly detection peformance.
    print(
        "neg norm=%f"
        % (np.linalg.norm(np.maximum(0, -Xopt[0]), "fro") + np.linalg.norm(np.maximum(0, -Xopt[1]), "fro"))
    )
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
