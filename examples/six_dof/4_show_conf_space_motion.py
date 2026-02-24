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

from aux import interpolate_equidistant
from examples.six_dof import P_EXAMPLE_6_DOF_DATA, P_EXAMPLE_6_DOF
from examples.six_dof.misc.vis import render_path, render_sphere, get_volumetric_corridor_mesh
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
    scene.delete_geometry(["x", "X"])
    X = Xs[cnt]
    qs, _ = np.split(X.T, 2, axis=1)
    q = qs[0]
    render_sphere(scene, q, r=0.05, color=[0, 0, 255], geom_name="x")
    render_path(scene, qs[:, :3], color=[0, 0, 255], geom_name="X")
    cnt = (cnt + 1) % Xs.shape[0]


world = DemoWorld()
q_s, q_g = world.get_demo_query()

man = world.man
path = world.get_demo_path()
path = interpolate_equidistant(path, delta=0.01)
path_radii = world.sdf(path[:, :3])

p_mesh = P_EXAMPLE_6_DOF / "cmesh.obj"
if p_mesh.exists():
    cmesh = trimesh.load(p_mesh)
else:
    print("missing cspace mesh...")
    cmesh = world.compute_cspace()
    cmesh.export(p_mesh)


p_corr = P_EXAMPLE_6_DOF / "corr.obj"
if p_corr.exists():
    corr = trimesh.load(p_corr)
else:
    print("missing cspace corridor mesh...")
    corr = get_volumetric_corridor_mesh(
        path[:, :3],
        path_radii,
        pitch=0.02,
        nr_ball_samples=50,
        verbose=True
    )
    corr = trimesh.smoothing.filter_humphrey(corr)
    corr.export(p_corr)


corr.visual.face_colors = [255, 0, 0, 100]
scene = trimesh.Scene(
    [
        cmesh,
        corr,
        trimesh.creation.axis()
    ]
)
render_sphere(scene, q_s[:3], r=0.05, color=[255, 0, 0])
render_sphere(scene, q_g[:3], r=0.05, color=[0, 255, 0])
render_path(scene, path[:, :3], color=[0, 0, 0])

scene.show(callback=callback)
