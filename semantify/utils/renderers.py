import cv2
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from omegaconf import ListConfig
from scipy.spatial.transform import Rotation
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesUV,
    BlendParams,
    Materials
)
from typing import Tuple, Optional, Union, List, Dict, Any, Literal
from semantify.utils.enums import TexturesPaths


class Pytorch3dRenderer:
    def __init__(
        self,
        model_type: Literal["smplx", "smal", "flame", "smpl"],
        device="cuda",
        dist: float = 0.5,
        elev: float = 0.0,
        azim: float = 0.0,
        img_size: Tuple[int, int] = (224, 224),
        texture_optimization: bool = False,
        use_tex: bool = False,
        background_color: Tuple[float, float, float] = (255.0, 255.0, 255.0),
    ):

        self.device = device
        self.background_color = background_color
        self.texture_optimization = texture_optimization
        tex_path = TexturesPaths[model_type.upper()].value if model_type is not None and use_tex else None
        self.tex_map = self._read_tex_img(tex_path)
        self.height, self.width = (
            img_size if isinstance(img_size, tuple) or isinstance(img_size, ListConfig) or isinstance(img_size, list) else (img_size, img_size)
        )

        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev)
        self.cameras = FoVPerspectiveCameras(znear=0.1, T=T, R=R, fov=30).to(self.device)
        lights = self.get_lights(self.device)
        materials = self.get_default_materials(self.device)
        blend_params = self.get_default_blend_params()
        self.shader = SoftPhongShader(
            device=self.device,
            cameras=self.cameras,
            lights=lights,
            materials=materials,
            blend_params=blend_params,
        )
        self._load_renderer()
        self._load_depth_rasterizer()

    def _load_renderer(self):
        self.raster_settings = RasterizationSettings(image_size=(self.height, self.width))
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=self.shader,
        )

    def _read_tex_img(self, tex_path: Optional[str] = None) -> Union[None, np.ndarray]:
        """
        This function returns a loaded texture image given a path.
        We assume working with SMPLX/FLAME default textures, hence it supports
        only .png and .npy files, such that the texutre in npy in BGR.
        """
        if tex_path is not None:
            if Path(tex_path).is_file() and Path(tex_path).exists():
                if tex_path.endswith(".png"):
                    return cv2.cvtColor(cv2.imread(tex_path), cv2.COLOR_BGR2RGB)
                elif tex_path.endswith(".npy"):
                    img = np.load(tex_path)
                    return img[..., ::-1]
                else:
                    raise ValueError("unrecognized texture type")
            else:
                raise ValueError("texture path does not exist")
        else:
            return None
        
    @staticmethod
    def get_texture(device, vt, ft, texture):
        verts_uvs = torch.as_tensor(vt, dtype=torch.float32, device=device)
        faces_uvs = torch.as_tensor(ft, dtype=torch.long, device=device)

        texture_map = torch.as_tensor(texture.copy(), device=device, dtype=torch.float32) / 255.0

        texture = TexturesUV(
            maps=texture_map[None],
            faces_uvs=faces_uvs[None],
            verts_uvs=verts_uvs[None],
        )
        return texture

    def _load_depth_rasterizer(self):
        self.depth_rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

    @staticmethod
    def get_lights(device):
        lights = PointLights(
            device=device,
            ambient_color=((0.8, 0.8, 0.8),),
            specular_color=((0.2, 0.2, 0.2),),
            location=[[0.0, 2.0, 2.0]],
        )
        return lights

    @staticmethod
    def get_default_materials(device):
        materials = Materials(device=device)
        return materials

    def get_default_blend_params(self):
        blend_params = BlendParams(sigma=1e-6, gamma=1e-6, background_color=self.background_color)
        return blend_params

    @staticmethod
    def rotate_3dmm_verts(
        verts: Union[torch.Tensor, np.ndarray], degrees: float, axis: Literal["x", "y", "z"]
    ) -> Meshes:
        convert_back_to_numpy = False
        if isinstance(verts, np.ndarray):
            convert_back_to_numpy = True
            verts = torch.tensor(verts).float()
        rotation_matrix = Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
        axis = 0 if verts.dim() == 2 else 1
        mesh_center = verts.mean(axis=axis)
        mesh_center_cloned = torch.tensor(mesh_center.clone().detach()).to(verts.device).float()
        rotation_matrix = torch.tensor(rotation_matrix).to(verts.device).float()
        verts = verts - mesh_center_cloned
        verts = verts @ rotation_matrix
        verts = verts + mesh_center_cloned
        if convert_back_to_numpy:
            verts = verts.cpu().numpy()
        return verts

    def render_mesh(
        self,
        verts: Union[torch.Tensor, np.ndarray] = None,
        faces: Union[torch.Tensor, np.ndarray] = None,
        vt: Optional[Union[torch.Tensor, np.ndarray]] = None,
        ft: Optional[Union[torch.Tensor, np.ndarray]] = None,
        mesh: Meshes = None,
        texture_color_values: Optional[torch.Tensor] = None,
        rotate_mesh: List[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        assert mesh is not None or (
            verts is not None and faces is not None
        ), "either mesh or verts and faces must be provided"
        if mesh is None:
            if rotate_mesh is not None:
                if isinstance(rotate_mesh, dict):
                    rotate_mesh = [rotate_mesh]
                for rot_action in rotate_mesh:
                    verts = self.rotate_3dmm_verts(verts, **rot_action)
            mesh = self.get_mesh(verts, faces, vt, ft, texture_color_values)

        rendered_mesh = self.renderer(mesh, cameras=self.cameras)
        return rendered_mesh

    def get_mesh(self, verts, faces, vt=None, ft=None, texture_color_values: torch.Tensor = None) -> Meshes:
        verts = torch.as_tensor(verts, dtype=torch.float32, device=self.device)
        faces = torch.as_tensor(faces, dtype=torch.long, device=self.device)
        if self.tex_map is not None:
            assert vt is not None and ft is not None, "vt and ft must be provided if texture is provided"
            texture = self.get_texture(self.device, vt, ft, self.tex_map)
        else:
            if len(verts.shape) == 2:
                verts = verts[None]

            if self.texture_optimization and texture_color_values is not None:
                texture = TexturesVertex(verts_features=texture_color_values)
            else:
                texture = TexturesVertex(
                    verts_features=torch.ones(*verts.shape, device=self.device)
                    * torch.tensor([0.7, 0.7, 0.7], device=self.device)
                )
        if len(verts.size()) == 2:
            verts = verts[None]
        if len(faces.size()) == 2:
            faces = faces[None]
        mesh = Meshes(verts=verts, faces=faces, textures=texture)
        return mesh

    def save_rendered_image(self, image, path, resize: bool = True):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().squeeze()
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
        if resize:
            image = cv2.resize(image, (512, 512))
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


class Open3dRenderer:
    def __init__(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        vt: torch.Tensor = None,
        ft: torch.Tensor = None,
        texture: str = None,
        light_on: bool = True,
        for_image: bool = True,
        img_size: Tuple[int, int] = (512, 512),
        paint_vertex_colors: bool = False,
    ):
        self.verts = verts
        self.faces = faces
        self.height, self.width = img_size
        self.paint_vertex_colors = paint_vertex_colors
        self.texture = cv2.cvtColor(cv2.imread(texture), cv2.COLOR_BGR2RGB) if texture is not None else None
        self.vt = vt
        self.ft = ft
        if self.vt is not None and self.ft is not None:
            uvs = np.concatenate([self.vt[self.ft[:, ind]][:, None] for ind in range(3)], 1).reshape(-1, 2)
            uvs[:, 1] = 1 - uvs[:, 1]
        else:
            uvs = None
        self.uvs = uvs
        self.for_image = for_image
        self.visualizer = o3d.visualization.Visualizer()
        self.default_zoom_value = 0.55
        self.default_y_rotate_value = 70.0
        self.default_up_translate_value = 0.3
        self.visualizer.create_window(width=self.width, height=self.height)
        opt = self.visualizer.get_render_option()
        if self.paint_vertex_colors:
            opt.background_color = np.asarray([255.0, 255.0, 255.0])
        else:
            opt.background_color = np.asarray([0.0, 0.0, 0.0])
        self.visualizer.get_render_option().light_on = light_on
        self.ctr = self.visualizer.get_view_control()
        self.ctr.set_zoom(self.default_zoom_value)
        self.ctr.camera_local_rotate(0.0, self.default_y_rotate_value, 0.0)
        self.ctr.camera_local_translate(0.0, 0.0, self.default_up_translate_value)
        self.mesh = self.get_initial_mesh()
        self.visualizer.add_geometry(self.mesh)
        self.mesh.compute_vertex_normals()

    def get_texture(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:

        if self.texture is not None and isinstance(self.texture, str):
            self.texture = cv2.cvtColor(cv2.imread(self.texture), cv2.COLOR_BGR2RGB)
        mesh.textures = [o3d.geometry.Image(self.texture)]
        mesh.triangle_uvs = o3d.utility.Vector2dVector(self.uvs)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh

    def get_initial_mesh(self) -> o3d.geometry.TriangleMesh:
        verts = (self.verts.squeeze() - self.verts.min()) / (self.verts.max() - self.verts.min())
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces.squeeze())
        if self.texture is not None:
            mesh = self.get_texture(mesh)

        if self.uvs is not None:
            mesh.triangle_uvs = o3d.utility.Vector2dVector(self.uvs)

        if self.paint_vertex_colors:
            mesh.paint_uniform_color([0.2, 0.8, 0.2])

        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(self.faces)), dtype=np.int32))
        return mesh

    def render_mesh(self, verts: torch.tensor = None, texture: np.array = None):

        if verts is not None:
            verts = (verts.squeeze() - verts.min()) / (verts.max() - verts.min())
            self.mesh.vertices = o3d.utility.Vector3dVector(verts.squeeze())
            self.visualizer.update_geometry(self.mesh)
        if texture is not None:
            self.texture = texture
            self.mesh = self.get_texture(self.mesh)
            self.visualizer.update_geometry(self.mesh)
        if self.for_image:
            self.visualizer.update_renderer()
            self.visualizer.poll_events()
        else:
            self.visualizer.run()

    def close(self):
        self.visualizer.close()

    def remove_texture(self):
        self.mesh.textures = []
        self.visualizer.update_geometry(self.mesh)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def zoom(self, zoom_value: float):
        self.ctr.set_zoom(zoom_value)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def rotate(self, y: float = None, x: float = 0.0, z: float = 0.0):
        if y is None:
            y = self.default_y_rotate_value
        self.ctr.camera_local_rotate(x, y, z)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def translate(self, right: float = 0.0, up: float = None):
        if up is None:
            up = self.default_up_translate_value
        self.ctr.camera_local_translate(0.0, right, up)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def reset_camera(self):
        self.ctr.set_zoom(self.default_zoom_value)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def save_mesh(self, path: str):
        o3d.io.write_triangle_mesh(path, self.mesh)
