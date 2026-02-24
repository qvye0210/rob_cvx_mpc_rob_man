# Copyright (c) 2025, ABB Schweiz AG
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn

import numpy as np


class SCDFNet(nn.Module):

    def __init__(
            self,
            config_dim,
            obstacle_dim,
            hidden_dim=256,
            nr_layers=3,
            dropouts=None
    ):
        feature_dim = config_dim + obstacle_dim
        self.obstacle_dim = obstacle_dim
        self.config_dim = config_dim
        dims = [feature_dim, ] + [hidden_dim] * nr_layers + [1]
        layer_dims = list(zip(dims[:-1], dims[1:]))
        layers = []
        dropouts = dropouts or []
        for i, (dim_in, dim_out) in enumerate(layer_dims):
            layers.append(nn.Linear(dim_in, dim_out))
            if i < len(dropouts):
                dropout_frac = dropouts[i]
                layers.append(nn.Dropout(dropout_frac))
            if i + 1 < len(layer_dims):
                layers.append(nn.ReLU())
        super().__init__()
        self.net = nn.Sequential(
            *layers
        )

    def forward(self, X):
        return self.net(X)

    def sdf(self, qs, dims, rots=None, reduce=True):
        is_single = len(qs.shape) == 1
        if is_single:
            qs = qs.reshape(1, -1)
        m, d = qs.shape
        n, l = dims.shape
        q_batch = qs.reshape((m, 1, d)).repeat(repeats=n, axis=1).reshape((m * n, d))
        dims_batch = dims.reshape((1, n, l)).repeat(repeats=m, axis=0).reshape((m * n, l))
        if rots is not None:
            rots_batch = rots.reshape(1, -1).repeat(repeats=m, axis=0).ravel()
            q_1 = (q_batch[:, 0] - rots_batch / np.pi)
            shift_trans = 1
            q_batch[:, 0] =  ((q_1 + shift_trans) % (2 * shift_trans) - shift_trans)
        X = np.hstack([dims_batch, q_batch])  # [m * n, ..]
        X_th = torch.tensor(X).type(torch.float32)
        with torch.no_grad():
            sdfs = self.forward(X_th).cpu().numpy().reshape((m, n))
            if reduce:
                sdfs = sdfs.min(axis=1)
        if is_single:
            return sdfs[0]
        else:
            return sdfs

    def compute_grad(self, q, dims, rots=None, reduce=True, return_dists=False):
        qs = np.tile(q, reps=(dims.shape[0], 1))
        if rots is not None:
            q_1 = (qs[:, 0] - rots / np.pi)
            shift_trans = 1
            qs[:, 0] = ((q_1 + shift_trans) % (2 * shift_trans) - shift_trans)
        X_q = torch.tensor(qs).type(torch.float32)
        X_q.requires_grad = True
        X_dims = torch.tensor(dims).type(torch.float32)
        X = torch.concatenate([X_dims, X_q], dim=1)
        sdists_obs = self.forward(X)     # TODO: add
        sdists, indxs_min = torch.min(sdists_obs, axis=0)
        grads, = torch.autograd.grad(sdists_obs.sum(), X_q)
        grads = grads.cpu().numpy()
        if reduce:
            grads = grads[indxs_min]
            sdists = sdists.detach().cpu().numpy().ravel()
        elif not reduce and return_dists:
            sdists = sdists_obs.detach().cpu().numpy().ravel()
        if return_dists:
            return grads, sdists
        else:
            return grads
