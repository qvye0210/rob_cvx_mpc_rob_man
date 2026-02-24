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


import pickle

import numpy as np
import trimesh

from examples.six_dof import P_EXAMPLE_6_DOF_DATA
from examples.six_dof.manipulator.man import DemoManipulator
from examples.six_dof.misc.vis import render_path
from examples.six_dof.world import DemoWorld

p_d = P_EXAMPLE_6_DOF_DATA / "dof_6_ef_0.02" / "motion" / "ft.pckl"


with p_d.open("rb") as fp:
    results = pickle.load(fp)

Xs = []
for result in results:
    X, U, S, r_p_0 = result.outputs
    Xs.append(X)

Xs = np.stack(Xs, axis=0)
cnt = 0

def callback(s):
    global cnt
    scene.delete_geometry(["X"])
    X = Xs[cnt]
    qs, _ = np.split(X.T, 2, axis=1)
    q = qs[0]
    man.update_scene(scene, q, s_data)
    ps = []
    for q in qs:
        T, = man.get_link_fk(q, links=["link_6"])
        ps.append(T[:3, -1])
    ps = np.vstack(ps)
    render_path(scene, ps, color=[0, 0, 255], geom_name="X")
    cnt = (cnt + 1) % Xs.shape[0]


world = DemoWorld()
q_s, q_g = world.get_demo_query()
man = DemoManipulator()

scene = trimesh.Scene(
    [
        world.s_obst.obst,
        trimesh.creation.axis()
    ]
)
man.add_to_scene(scene, q_s, color=[255, 0, 0, 100])
man.add_to_scene(scene, q_g, color=[0, 255, 0, 100])
s_data = man.add_to_scene(scene, geom_name_suffix="t")
scene.show(callback=callback)

