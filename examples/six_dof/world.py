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


import pathlib

import numpy as np
import trimesh

from aux import interpolate_equidistant
from examples.six_dof.manipulator.man import DemoManipulator
from examples.six_dof.learn.load_exp_models import load_model_from_exp_name
from examples.six_dof.misc.vis import compute_cspace_mesh


P_CURR = pathlib.Path(__file__).parent


class DemoStaticObstacles:

    def __init__(self):
        p_s = np.r_[0.3, 0.0, 0.0]
        p_e = np.r_[0.3, 0.0, 0.3]
        N = 4
        s = np.linspace(0, 1, N)
        s = s[:, None]
        r = 0.075
        centers = p_s * (1-s) + p_e * s
        self.dims = np.c_[centers, r * np.ones(N)]
        self.obst = [trimesh.creation.icosphere(radius=r).apply_translation(c) for *c, r in self.dims]
        self.cm_obst = trimesh.collision.CollisionManager()
        for i, o in enumerate(self.obst):
            self.cm_obst.add_object(name=f"obst_{i}", mesh=o)


class DemoWorld:

    def __init__(self):
        self.s_obst = DemoStaticObstacles()
        self.man = DemoManipulator()
        self.model = load_model_from_exp_name("pretrained")
        z = np.r_[np.pi, np.pi / 2, np.pi * 0.8, np.pi * np.ones(3, )]
        self.lims = np.c_[-z, z].T

    def is_collision_free_gt(self, q):
        return self.man.is_collision_free(q, cm_obst=self.s_obst.cm_obst)

    def compute_cspace(self):
        cmesh = compute_cspace_mesh(
            self.sdf,
            pitch=0.075,
            batch_size=int(1e4),
            nr_points=100,
            grid_l_u=self.lims[:, :3],
            verbose=True
        )
        cmesh = trimesh.smoothing.filter_humphrey(cmesh)
        return cmesh


    def get_demo_query(self):
        q_s = np.r_[-0.6, np.pi / 4, np.pi / 2, np.zeros(3, )]
        q_g = np.r_[0.6, np.pi / 4, np.pi / 2, np.zeros(3, )]
        return q_s, q_g

    def get_demo_path(self):
        q_s, q_g = self.get_demo_query()
        path = np.vstack([
            q_s,
            np.r_[-0.6, 0, np.pi / 3, np.zeros(3, )],
            np.r_[0.6, 0, np.pi / 3, np.zeros(3, )],
            q_g
        ])
        return path

    def sdf(self, q):
        return self.model.sdf(q, dims=self.s_obst.dims)
