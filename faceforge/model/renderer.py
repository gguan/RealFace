"""
Differentiable mesh renderer (PyTorch3D-based).
PyTorch3D rasterizer ops auto-fallback to CPU on MPS (handled here).
"""

import torch
from dataclasses import dataclass
from typing import Optional

from loguru import logger

try:
    from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        TexturesVertex,
        PointLights,
        FoVOrthographicCameras,
    )
    from pytorch3d.structures import Meshes
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False


@dataclass
class RenderOutput:
    image:        torch.Tensor  # (B, H, W, 4) RGBA float [0,1]
    zbuf:         torch.Tensor  # (B, H, W, 1) depth
    pix_to_face:  torch.Tensor  # pixel->face mapping (used for mask)


class DifferentiableRenderer:
    """
    Differentiable renderer using PyTorch3D.
    MPS note: rasterizer fallback to CPU automatically on MPS.
    """

    def __init__(self, image_size: int = 512, device=None):
        """
        Initialize rasterizer and shader.
        image_size: render resolution. Mac recommended: 256 to save memory.
        """
        from faceforge.utils.device import get_device
        self.image_size = image_size
        self.device = device or get_device()
        self._raster_device = torch.device("cpu") if self.device.type == "mps" else self.device

        if HAS_PYTORCH3D:
            self._raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            self.rasterizer = MeshRasterizer(raster_settings=self._raster_settings)
            logger.info(
                f"[renderer] PyTorch3D available. Device: {self.device}, "
                f"raster_device: {self._raster_device}"
            )
        else:
            logger.warning(
                "[renderer] PyTorch3D not available — rendering disabled, returning dummy outputs"
            )

    def _rasterize_with_fallback(self, mesh, cameras):
        """Rasterize step falls back to CPU on MPS."""
        if self.device.type == "mps":
            mesh_cpu = mesh.to("cpu")
            cameras_cpu = cameras.to("cpu")
            raster_cpu = self.rasterizer(mesh_cpu, cameras=cameras_cpu)
            # fragments can stay on CPU for shader too
            return raster_cpu
        return self.rasterizer(mesh, cameras=cameras)

    def render(
        self,
        vertices:      torch.Tensor,           # (B, V, 3)
        faces:         torch.Tensor,           # (F, 3)
        cameras,                               # PerspectiveCameras or similar
        lights:        Optional = None,
        vertex_colors: Optional[torch.Tensor] = None,  # (B, V, 3)
    ) -> RenderOutput:
        """Differentiable render, returns RGBA image."""
        if not HAS_PYTORCH3D:
            B = vertices.shape[0]
            H = W = self.image_size
            return RenderOutput(
                image=torch.zeros(B, H, W, 4, device=self.device),
                zbuf=torch.zeros(B, H, W, 1, device=self.device),
                pix_to_face=torch.full(
                    (B, H, W, 1), -1, dtype=torch.long, device=self.device
                ),
            )

        B = vertices.shape[0]

        # Build vertex colors/textures
        if vertex_colors is None:
            # Default: gray mesh
            vertex_colors = torch.ones_like(vertices) * 0.5
        textures = TexturesVertex(verts_features=vertex_colors)

        # Build Meshes object — faces must be (F,3) repeated per batch
        faces_batch = faces.unsqueeze(0).expand(B, -1, -1)  # (B, F, 3)
        meshes = Meshes(
            verts=list(vertices),
            faces=list(faces_batch),
            textures=textures,
        )

        # Setup lights
        if lights is None:
            lights = PointLights(
                device=self._raster_device,
                location=[[0.0, 0.0, -2.0]],
            )

        # Rasterize (with MPS fallback)
        fragments = self._rasterize_with_fallback(meshes, cameras)

        # Shade
        shader = SoftPhongShader(
            device=self._raster_device,
            cameras=cameras.to(self._raster_device),
            lights=lights,
        )
        images = shader(fragments, meshes.to(self._raster_device))  # (B, H, W, 4)

        return RenderOutput(
            image=images.to(self.device),
            zbuf=fragments.zbuf.to(self.device),
            pix_to_face=fragments.pix_to_face.to(self.device),
        )

    def build_cameras(
        self,
        cam_params: torch.Tensor,  # (B, 3) [scale, tx, ty] weak-perspective
        image_size: int,
    ):
        """Build FoVOrthographicCameras from weak-perspective camera params."""
        if not HAS_PYTORCH3D:
            return None

        # cam_params: (B, 3) = [scale, tx, ty]
        scale = cam_params[:, 0:1]  # (B, 1)
        tx = cam_params[:, 1:2]
        ty = cam_params[:, 2:3]

        cameras = FoVOrthographicCameras(
            device=self._raster_device,
            znear=0.001,
            zfar=10.0,
            max_y=1.0 / scale.mean().item(),
            min_y=-1.0 / scale.mean().item(),
            max_x=1.0 / scale.mean().item(),
            min_x=-1.0 / scale.mean().item(),
            T=torch.cat([tx, ty, torch.ones_like(tx)], dim=1),
        )
        return cameras

    def project_points(
        self,
        points3d:  torch.Tensor,   # (B, N, 3)
        cameras,
    ) -> torch.Tensor:             # (B, N, 2) pixel coordinates
        """Project 3D points to 2D."""
        if cameras is None or not HAS_PYTORCH3D:
            # Fallback: simple orthographic projection
            return points3d[..., :2]

        pts = cameras.transform_points(points3d.to(self._raster_device))  # NDC
        # Convert NDC to pixel coords in [0, image_size]
        px = (pts[..., 0] + 1) * self.image_size / 2
        py = (1 - pts[..., 1]) * self.image_size / 2  # flip y
        return torch.stack([px, py], dim=-1).to(self.device)

    def extract_face_mask(
        self,
        render_output: RenderOutput,
    ) -> torch.Tensor:             # (B, H, W) bool
        """Extract face region mask from render output."""
        # pix_to_face: (B, H, W, 1) — -1 means background
        mask = (render_output.pix_to_face[..., 0] >= 0)  # (B, H, W) bool
        return mask
