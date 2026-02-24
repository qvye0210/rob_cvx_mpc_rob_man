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

import numpy as np
import trimesh
from trimesh.path import Path3D
from trimesh.path.entities import Line
from trimesh.voxel.ops import points_to_marching_cubes

import tqdm


def compute_cspace_mesh(
        sdf,
        pitch=0.1,
        batch_size=None,
        nr_points=32,
        grid_l_u=None,
        verbose=False,
        mcube_trans_scale=None,
        pad_confs=None
):
    G = nr_points
    if grid_l_u is None:
        q = np.linspace(-np.pi, np.pi, G)
        qs = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3)
    else:
        M = grid_l_u.shape[1]
        q = np.linspace(grid_l_u[0], grid_l_u[1], G)
        qs = np.stack(np.meshgrid(*q.T, indexing="ij"), axis=-1).reshape(-1, 3)
    if pad_confs is not None:
        qs = np.c_[qs, pad_confs[None].repeat(qs.shape[0], axis=0)]
    if batch_size is None:
        sdistance = sdf(qs)
    else:
        M = qs.shape[-1]
        qs = qs.reshape((-1, batch_size, M))
        if verbose:
            tbar = tqdm.tqdm(qs, desc="Computing C-space")
        else:
            tbar = qs
        sdistance = np.hstack([sdf(qs_) for qs_ in tbar])
        qs = qs.reshape((-1, M))
    mask = sdistance < 0
    qs_c = qs[mask][:, :3]
    if mcube_trans_scale is not None:
        trans, scale = mcube_trans_scale
        qs_c = (qs_c - trans) / scale
    mcubes = trimesh.voxel.ops.points_to_marching_cubes(qs_c, pitch=pitch)
    return mcubes



def render_sphere(scene, c, r, color, node_name=None, geom_name=None):
    sph = trimesh.creation.icosphere(radius=r)
    sph.apply_translation(c[:3])
    sph.visual.face_colors = color
    scene.add_geometry(sph, node_name=node_name, geom_name=geom_name)


def render_path(scene, path, color=[0, 0, 255], node_name=None, geom_name=None):
    scene.add_geometry(Path3D(
        entities=[Line(np.r_[0:path.shape[0]], color=color)],
        vertices=path
    ), node_name=node_name, geom_name=geom_name
    )


def get_volumetric_corridor_mesh(
        path_centers,
        path_radii,
        pitch,
        nr_ball_samples=20,
        verbose=False,
):
    z = np.linspace(-1, 1, nr_ball_samples)
    xys = np.stack(np.meshgrid(z, z, z, indexing="ij"), axis=-1).reshape(-1, 3)
    all_pnts = []
    stuff = zip(path_centers, path_radii)
    tbar = tqdm.tqdm(stuff, total=path_radii.size) if verbose else stuff
    for c, r in tbar:
        stuff = xys * r
        stuff_ = stuff[np.linalg.norm(stuff, axis=1) < r] + c
        all_pnts.append(stuff_)
    points = np.vstack(all_pnts)
    mesh = points_to_marching_cubes(points, pitch=pitch)
    return mesh