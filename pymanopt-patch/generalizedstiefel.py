# Copyright (c) 2019-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2015-2016, Pymanopt Developers.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-3-Clause
# All rights reserved.
#
# Modified by Anoop Cherian, cherian@merl.com

from __future__ import division

import numpy as np
from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multiprod, multisym, multitransp
from scipy.linalg import expm


class GeneralizedStiefel(Manifold):
    """
    Factory class for the Stiefel manifold. Instantiation requires the
    dimensions n, p to be specified. Optional argument k allows the user to
    optimize over the product of k Stiefels.

    Elements are represented as n x p matrices (if k == 1), and as k x n x p
    matrices if k > 1 (Note that this is different to manopt!).
    """

    def __init__(self, n, p, K, k=1):
        self._n = n
        self._p = p
        self._k = k
        if k == 1:
            self._K = K
            self._invK = np.linalg.inv(K)
        else:
            repmat = lambda kk: np.concatenate([kk[np.newaxis, :, :] for i in range(k)], axis=0)
            self._K = repmat(K)
            self._invK = repmat(np.linalg.inv(K))

        # Check that n is greater than or equal to p
        if n < p or p < 1:
            raise ValueError("Need n >= p >= 1. Values supplied were n = %d " "and p = %d." % (n, p))
        if k < 1:
            raise ValueError("Need k >= 1. Value supplied was k = %d." % k)

        # we also need to check that K is symm. and is spd. but will skip for now.
        if k == 1:
            name = "Generalized Stiefel manifold St(%d, %d)" % (n, p)
        elif k >= 2:
            name = "Product Generalized Stiefel manifold St(%d, %d)^%d" % (n, p, k)
        self.dimension = int(k * (n * p - p * (p + 1) / 2))
        # super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self._p * self._k)

    # inner product is trace(G'KH)
    def inner(self, X, G, H):
        # Inner product (Riemannian metric) on the tangent space
        # For the stiefel this is the Frobenius inner product.
        return np.tensordot(G, multiprod(self._K, H), axes=G.ndim)

    def proj(self, X, U):
        KX = multiprod(self._K, X)
        return U - multiprod(X, multisym(multiprod(multitransp(KX), U)))

    def egrad2rgrad(self, X, egrad):
        egrad_scaled = multiprod(self._invK, egrad)
        return egrad_scaled - multiprod(X, multisym(multiprod(multitransp(X), egrad)))

    # TODO(nkoep): Implement the weingarten map instead.
    def ehess2rhess(self, X, egrad, ehess, H):
        egraddot = ehess
        Xdot = H
        egrad_scaleddot = multiprod(self._invK, egraddot)
        rgraddot = (
            egrad_scaleddot
            - multiprod(Xdot, multisym(multiprod(multitransp(X), egrad)))
            - multiprod(multisym(multiprod(multitransp(Xdot), egrad)))
            - multiprod(multisym(multiprod(multitransp(Xdot), egraddot)))
        )
        return self.proj(X, rgraddot)

    # Retract to the Stiefel using the qr decomposition of X + G.
    def retr(self, X, G, t=1.0):
        return self.guf(X + t * G)

    def norm(self, X, G):
        # Norm on the tangent space of the Stiefel is simply the Euclidean
        # norm.
        return np.linalg.norm(G)

    # Generate random Stiefel point using qr of random normally distributed
    # matrix.
    def rand(self):
        if self._k == 1:
            X = self.guf(np.random.randn(self._n, self._p))
            return X

        X = np.zeros((self._k, self._n, self._p))
        for i in range(self._k):
            X[i], _ = self.guf(np.random.randn(self._n, self._p))
        return X

    def randvec(self, X):
        U = np.random.randn(*np.shape(X))
        U = self.proj(X, U)
        U = U / np.linalg.norm(U)
        return U

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def exp(self, X, U):
        # TODO: Simplify these expressions.
        return self.retr(X, U)

    def zerovec(self, X):
        if self._k == 1:
            return np.zeros((self._n, self._p))
        return np.zeros((self._k, self._n, self._p))

    def guf(self, U):
        u, _, vt = np.linalg.svd(U, full_matrices=False)
        ssquare, q = np.linalg.eig(np.matmul(u.transpose(), np.matmul(self._K, u)))
        qsinv = np.matmul(q, np.diag(1.0 / np.sqrt(ssquare)))
        X = np.matmul(u, np.matmul(np.matmul(qsinv, np.transpose(q)), vt))
        return X
