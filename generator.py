"""
Hunyuan3D 2 Mini Fast + Paint PBR v2.1
=======================================
- Shape  : tencent/Hunyuan3D-2mini  (hunyuan3d-dit-v2-mini-fast)
- Texture: tencent/Hunyuan3D-2.1    (hunyuan3d-paintpbr-v2-1)
"""

import io
import os
import random
import sys
import tempfile
import time
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional
from PIL import Image
from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_HF_REPO_ID      = "tencent/Hunyuan3D-2mini"
_SUBFOLDER       = "hunyuan3d-dit-v2-mini-fast"
_GITHUB_ZIP      = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"

# ── Texture model (PBR v2.1) ──────────────────────────────────────────────────
_PAINT_HF_REPO   = "tencent/Hunyuan3D-2.1"
_PAINT_SUBFOLDER = "hunyuan3d-paintpbr-v2-1"


class Hunyuan3DMiniFastGenerator(BaseGenerator):

    MODEL_ID     = "hunyuan3d-mini-fast"
    DISPLAY_NAME = "Hunyuan3D 2 Mini Fast"
    VRAM_GB      = 6

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def is_downloaded(self) -> bool:
        subfolder  = self.download_check if self.download_check else _SUBFOLDER
        model_dir  = self.model_dir / subfolder
        return model_dir.exists() and (model_dir / "model.fp16.safetensors").exists()

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_downloaded():
            self._download_weights()
        self._ensure_hy3dgen()

        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        subfolder = self.download_check if self.download_check else _SUBFOLDER

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(self.model_dir),
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
            dtype=dtype,
        )
        self._model = pipeline

    def unload(self) -> None:
        super().unload()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        num_steps      = int(params.get("num_inference_steps", 10))
        enable_texture = bool(params.get("enable_texture", False))
        octree_res     = int(params.get("octree_resolution", 380))
        guidance_scale = float(params.get("guidance_scale", 5.5))
        tex_resolution = int(params.get("texture_resolution", 1024))
        seed           = int(params.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        self._report(progress_cb, 5, "Removing background…")
        image = self._preprocess(image_bytes)
        self._check_cancelled(cancel_event)

        shape_end = 68 if enable_texture else 82
        self._report(progress_cb, 12, "Generating 3D shape…")

        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, shape_end, "Generating 3D shape…", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            with torch.no_grad():
                generator = torch.Generator().manual_seed(seed)
                outputs = self._model(
                    image=image,
                    num_inference_steps=num_steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=4000,
                    generator=generator,
                    output_type="trimesh",
                )
            mesh = outputs[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        if enable_texture:
            self._report(progress_cb, 70, "Freeing VRAM for texture model…")
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._check_cancelled(cancel_event)
            mesh = self._run_texture(mesh, image, tex_resolution, progress_cb)
        
        self._report(progress_cb, 96, "Exporting GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))
        self._report(progress_cb, 100, "Done")
        return path

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        import rembg
        img = Image.open(io.BytesIO(image_bytes))
        try:
            return rembg.remove(img).convert("RGBA")
        except Exception:
            session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            return rembg.remove(img, session=session).convert("RGBA")

    def _run_texture(self, mesh, image: Image.Image, resolution: int, progress_cb=None):
        import torch
        self._check_texgen_extensions()
        self._report(progress_cb, 73, "Downloading PBR texture model (v2.1)…")
        self._ensure_paint_weights()
        self._report(progress_cb, 80, "Loading texture model…")

        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        paint_dir = self.model_dir / "_paint_weights"
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            str(paint_dir), subfolder=_PAINT_SUBFOLDER
        )
        paint_pipeline.config.render_size  = resolution
        paint_pipeline.config.texture_size = resolution
        paint_pipeline.render = MeshRender(
            default_resolution=resolution, texture_size=resolution
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            image.save(tmp.name)
            tmp.close()
            self._report(progress_cb, 85, "Generating PBR textures…")
            with torch.no_grad():
                result = paint_pipeline(mesh, image=tmp.name)
        finally:
            os.unlink(tmp.name)

        del paint_pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result[0] if isinstance(result, (list, tuple)) else result

    def _check_texgen_extensions(self) -> None:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: F401
        except (ImportError, OSError) as exc:
            base = self.model_dir / "_hy3dgen" / "hy3dgen" / "texgen"
            raise RuntimeError(
                "C++ texture extensions not compiled.\n"
                f"Run:\n"
                f"  cd \"{base / 'custom_rasterizer'}\" && python setup.py install\n"
                f"  cd \"{base / 'differentiable_renderer'}\" && python setup.py install\n"
                f"Error: {exc}"
            ) from exc

    def _ensure_paint_weights(self) -> None:
        paint_dir = self.model_dir / "_paint_weights"
        marker    = paint_dir / _PAINT_SUBFOLDER / "model_index.json"
        if marker.exists():
            return
        from huggingface_hub import snapshot_download
        print(f"[Hunyuan3D] Downloading PBR paint model ({_PAINT_HF_REPO}/{_PAINT_SUBFOLDER})…")
        snapshot_download(
            repo_id=_PAINT_HF_REPO,
            local_dir=str(paint_dir),
            allow_patterns=[f"{_PAINT_SUBFOLDER}/**"],
        )
        print("[Hunyuan3D] PBR paint model downloaded.")

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=[
                "hunyuan3d-dit-v2-mini/",
                "hunyuan3d-dit-v2-mini-turbo/",
                "hunyuan3d-vae-v2-mini-turbo/",
                "hunyuan3d-vae-v2-mini-withencoder/",
                "*.md", "LICENSE", "NOTICE", ".gitattributes",
            ],
        )

    def _ensure_hy3dgen(self) -> None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
            return
        except ImportError:
            pass
        src_dir = self.model_dir / "_hy3dgen"
        if not (src_dir / "hy3dgen").exists():
            self._download_hy3dgen(src_dir)
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

    def _download_hy3dgen(self, dest: Path) -> None:
        import urllib.request
        dest.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=180) as resp:
            data = resp.read()
        prefix = "Hunyuan3D-2-main/hy3dgen/"
        strip  = "Hunyuan3D-2-main/"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel    = member[len(strip):]
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id": "num_inference_steps",
                "label": "Quality",
                "type": "select",
                "default": 10,
                "options": [
                    {"value": 5,  "label": "Fast"},
                    {"value": 10, "label": "Balanced"},
                    {"value": 20, "label": "High"},
                ],
            },
            {
                "id": "octree_resolution",
                "label": "Mesh Resolution",
                "type": "select",
                "default": 380,
                "options": [
                    {"value": 256, "label": "Low"},
                    {"value": 380, "label": "Medium"},
                    {"value": 512, "label": "High"},
                ],
            },
            {
                "id": "guidance_scale",
                "label": "Guidance Scale",
                "type": "float",
                "default": 5.5,
                "min": 1.0,
                "max": 10.0,
                "step": 0.5,
            },
            {
                "id": "seed",
                "label": "Seed",
                "type": "int",
                "default": -1,
                "min": -1,
                "max": 4294967295,
            },
            {
                "id": "enable_texture",
                "label": "Enable Texture (PBR)",
                "type": "bool",
                "default": False,
                "tooltip": "Generate PBR textures with Hunyuan3D-Paint v2.1 (~7 GB extra download on first run).",
            },
            {
                "id": "texture_resolution",
                "label": "Texture Resolution",
                "type": "select",
                "default": 1024,
                "options": [
                    {"value": 512,  "label": "512"},
                    {"value": 1024, "label": "1024"},
                    {"value": 2048, "label": "2048"},
                ],
            },
        ]
