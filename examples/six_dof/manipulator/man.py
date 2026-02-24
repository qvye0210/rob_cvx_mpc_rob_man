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

import trimesh
import numpy as np
import pinocchio as pin
from trimesh.collision import CollisionManager

PATH_ROOT = pathlib.Path(__file__).parent


z = np.r_[np.pi, np.pi/2, np.pi*0.8, 0.0, 0.0, 0.0]
LIMS = np.c_[z, -z].T

class DemoManipulator:

    def __init__(self):
        p_urdf = PATH_ROOT / "demo.urdf"
        link_names = [f"link_{i}" for i in range(7)]
        links_coll = ["link_2", "link_3"]
        self.model = pin.buildModelFromUrdf(str(p_urdf))
        self.data = self.model.createData()
        self.config_dim = self.model.nq

        self.link_names = link_names
        self.link_ids_maps = {ln: self.model.getFrameId(ln) for ln in self.link_names}
        self.link_vizs = []
        self.vis_meshes = []
        self.p_meshes = p_urdf.parent / "meshes"
        self.links_coll = []
        self.coll_meshes = []
        self.cm: trimesh.collision.CollisionManager = trimesh.collision.CollisionManager()

        self.cm_selfs = []
        self.links_coll_self = []
        self.coll_meshes_self = {}

        if links_coll:
            self.set_collision_meshes(
                self.p_meshes, links_coll
            )
        self.self_collision = False

    def set_collision_meshes(self, p_meshes, links_coll, links_mesh_map=None):
        p_collision = p_meshes
        self.links_coll = links_coll
        self.coll_meshes = []
        for i, l_name in enumerate(self.links_coll):
            if links_mesh_map is not None:
                mesh_name = f"{links_mesh_map.get(l_name, l_name)}.obj"
            else:
                mesh_name = f"{l_name}.obj"
            if (p_collision / mesh_name).exists():
                mesh = trimesh.load_mesh(p_collision / mesh_name)
                self.cm.add_object(name=f"obj_{i}", mesh=mesh)
                self.coll_meshes.append(mesh)

    def set_self_collision_pairs(self, p_meshes, links_coll_self_pairs, links_mesh_map=None):
        p_collision = p_meshes
        links_mesh_map = links_mesh_map or {}
        for (l_1, l_2) in links_coll_self_pairs:
            cm_self = CollisionManager()
            for i, l_name in enumerate((l_1, l_2)):
                if l_name not in self.coll_meshes_self:
                    mesh_name = f"{links_mesh_map.get(l_name, l_name)}.obj"
                    mesh = trimesh.load_mesh(p_collision / mesh_name)
                    self.coll_meshes_self[l_name] = mesh
                else:
                    mesh = self.coll_meshes_self[l_name]
                cm_self.add_object(name=f"obj_{i}", mesh=mesh)
            self.cm_selfs.append(cm_self)
            self.links_coll_self.extend([l_1, l_2])

    def is_collision_free_self(self, q):
        if not self.self_collision:
            return True
        fks = self.get_link_fk(
            q=q, links=self.links_coll_self
        )
        coll = False
        for i, cm_self in enumerate(self.cm_selfs):
            k = i * 2
            for j, pose in enumerate(fks[k:k + 2]):
                cm_self.set_transform(f"obj_{j}", pose)
            coll = coll or cm_self.in_collision_internal()
        return not coll

    def is_collision_free(self, q, obst=None, cm_obst=None, return_extra=False):
        fk = self.get_link_fk(
            q=q, links=self.links_coll
        )
        for i, pose in enumerate(fk):
            self.cm.set_transform(f"obj_{i}", pose)
        if obst is not None:
            data = self.cm.in_collision_single(obst, return_names=return_extra, return_data=return_extra)
        elif cm_obst is not None:
            data = self.cm.in_collision_other(cm_obst, return_names=return_extra, return_data=return_extra)
        else:
            data = False
        in_coll_internal = False
        if return_extra:
            in_coll, objs_in_collision, contact_data = data
            in_coll = in_coll or in_coll_internal
            return not in_coll, objs_in_collision, contact_data
        else:
            in_coll = data
            in_coll = in_coll or in_coll_internal
            return not in_coll

    def get_link_fk(self, q=None, links=None, **kwargs):
        q_ = np.zeros(self.config_dim, )
        if q is not None:
            q_[:q.size] = q
        link_names = links or self.link_names
        pin.forwardKinematics(self.model, self.data, q_)
        pin.updateFramePlacements(self.model, self.data)
        return [self.data.oMf[self.link_ids_maps[f_name]].homogeneous for f_name in link_names]

    def smallest_distance(self, q, obst=None, cm_obst=None, **kwargs):
        fk = self.get_link_fk(q, links=self.links_coll)
        for i, pose in enumerate(fk):
            self.cm.set_transform(f"obj_{i}", pose)
        if obst is not None:
            return self.cm.min_distance_single(obst, **kwargs)
        else:
            return self.cm.min_distance_other(cm_obst, **kwargs)

    def add_to_scene(
            self,
            scene,
            q=None,
            node_name_suffix=None,
            color=None,
            geom_name_suffix=None,
            collision=True,
            origin=False
    ):
        if collision:
            links = self.links_coll
            fk = self.get_link_fk(q, links=links)
            outs = zip(links, fk, self.coll_meshes)
        else:
            links = self.link_vizs
            fk = self.get_link_fk(q, links=links)
            outs = zip(links, fk, self.vis_meshes)
        geom_name = None
        node_name = None
        scene_data = []
        for i, (l_name, f, m) in enumerate(outs):
            m_ = m
            if node_name_suffix:
                node_name = f"{node_name_suffix}_{i}"
            if color:
                m_ = m_.copy()
                m_.visual.face_colors = color
            if geom_name_suffix:
                geom_name = f"{geom_name_suffix}_{i}"
            scene_data.append((l_name, geom_name))
            scene.add_geometry(m_, transform=f, node_name=node_name, geom_name=geom_name)
        if origin:
            scene.add_geometry(trimesh.creation.axis())
        return scene_data

    def update_scene(self, scene, q, scene_data):
        links = [l for l, g in scene_data]
        fk = self.get_link_fk(q, links=links)
        graph = scene.graph
        for T, (l, geom_name) in zip(fk, scene_data):
            graph.update(frame_to=geom_name, frame_from="world", matrix=T)

    def get_geometries(self, q, links=None):
        fk = self.get_link_fk(q, links=links)
        return list(zip(fk, self.coll_meshes))


if __name__ == "__main__":
    man = DemoManipulator()
    scene = trimesh.Scene()
    q = np.r_[0, 0., 0.0]
    man.add_to_scene(scene, q=q, origin=True)
    fks = man.get_link_fk(q, links=[f"link_{i}" for i in range(7)])
    for T in fks:
        scene.add_geometry(trimesh.creation.axis(), transform=T)
    scene.show()
