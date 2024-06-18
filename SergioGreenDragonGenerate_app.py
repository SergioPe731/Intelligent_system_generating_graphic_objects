#define your first application
#!pip install gradio
from __future__ import annotations

import subprocess

import gradio as gr
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from gradio import themes
import time

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log
import logging
import tempfile
import time
import rembg
from PIL import Image
from functools import partial

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
import argparse
from tripo_app import *

#Old Standard TT
#Philosopher
#Literata
#Alice
#Playpen Sans
#Averia Serif Libre
#Rye
#Gabriela
#Elsie
#Caprasimo
#Balthazar
#Trade Winds
#New Rocker

import imageio
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_glb
from src.utils.infer_util import remove_background, resize_foreground, images_to_video

import tempfile
from huggingface_hub import hf_hub_download
def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def images_to_video(images, output_path, fps=30):
    # images: (N, C, H, W)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='h264')

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

def preprocess_instantmesh(input_image, do_remove_background):
    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    return input_image

def generate_mvs(input_image, sample_steps, sample_seed):
    seed_everything(sample_seed)

    # sampling
    generator = torch.Generator(device=device0)
    z123_image = pipeline(
        input_image,
        num_inference_steps=sample_steps,
        generator=generator,
    ).images[0]

    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image)  # (960, 640, 3)
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())

    return z123_image, show_image


def make_mesh(mesh_fpath, planes):
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")

    with torch.no_grad():
        # get mesh

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]

        save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)
        save_obj(vertices, faces, vertex_colors, mesh_fpath)

        print(f"Mesh saved to {mesh_fpath}")

    return mesh_fpath, mesh_glb_fpath


def make3d(images):
    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)  # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device1)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device1)

    images = images.unsqueeze(0).to(device1)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    mesh_fpath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
    print(mesh_fpath)
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get video
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 384

        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if IS_FLEXICUBES:
                frame = model.forward_geometry(
                    planes,
                    render_cameras[:, i:i + chunk_size],
                    render_size=render_size,
                )['img']
            else:
                frame = model.synthesizer(
                    planes,
                    cameras=render_cameras[:, i:i + chunk_size],
                    render_size=render_size,
                )['images_rgb']
            frames.append(frame)
        frames = torch.cat(frames, dim=1)

        images_to_video(
            frames[0],
            video_fpath,
            fps=30,
        )

        print(f"Video saved to {video_fpath}")

    mesh_fpath, mesh_glb_fpath = make_mesh(mesh_fpath, planes)

    return video_fpath, mesh_fpath, mesh_glb_fpath

_HEADER_ = '''
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/TencentARC/InstantMesh' target='_blank'><b>InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models</b></a></h2>

**InstantMesh** is a feed-forward framework for efficient 3D mesh generation from a single image based on the LRM/Instant3D architecture.

Code: <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>GitHub</a>. Techenical report: <a href='https://arxiv.org/abs/2404.07191' target='_blank'>ArXiv</a>.

❗️❗️❗️**Important Notes:**
- Our demo can export a .obj mesh with vertex colors or a .glb mesh now. If you prefer to export a .obj mesh with a **texture map**, please refer to our <a href='https://github.com/TencentARC/InstantMesh?tab=readme-ov-file#running-with-command-line' target='_blank'>Github Repo</a>.
- The 3D mesh generation results highly depend on the quality of generated multi-view images. Please try a different **seed value** if the result is unsatisfying (Default: 42).
'''

_CITE_ = r"""
If InstantMesh is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/TencentARC/InstantMesh?style=social)](https://github.com/TencentARC/InstantMesh)
---
📝 **Citation**

If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

📋 **License**

Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/TencentARC/InstantMesh/blob/main/LICENSE) for details.

📧 **Contact**

If you have any questions, feel free to open a discussion or contact us at <b>bluestyle928@gmail.com</b>.
"""

def reconstruction_object(inp_file):
    new_obj=''
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    size_work_memory = torch.cuda.mem_get_info()[0]

    if torch.cuda.is_available() and size_work_memory>=12884901888:
        new_obj=run_instantmesh(inp_file)
    else:
        new_obj = run_tripo(inp_file)

    return new_obj

seafoam = gr.themes.Glass(
    primary_hue="emerald",
    secondary_hue="indigo",
    font=[gr.themes.GoogleFont('Elsie'), gr.themes.GoogleFont('Alice'), 'Arial', 'source-sans-pro'],
).set(
    background_fill_primary='*neutral_300',
    body_background_fill="*primary_200",
    body_text_color="*secondary_hue_800",
    body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
    button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
    button_primary_text_color="white",
    button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
    slider_color="*secondary_300",
    slider_color_dark="*secondary_600",
    block_title_text_weight="600",
    block_border_width="3px",
    block_shadow="*shadow_drop_lg",
    button_shadow="*shadow_drop_lg",
    button_large_padding="32px",
    )
def run_tripo(inp_file):
    # запуск скрипта run.py
    os.system(f'python tripo_run.py {inp_file} --output-dir output_tripo/')
    # вернуть путь до 3D результата (заменить пути на свои)
    return 'C:\\Users\\user\\Intelligent_system_generating_graphic_objects\\output_tripo\\0\\mesh.obj'

def run_instantmesh(inp_file):
    # запуск скрипта run.py
    os.system(f'python Instantmesh_run.py configs/instant-mesh-large.yaml {inp_file} --output_path output_instantmesh --save_video')
    # вернуть путь до 3D результата (заменить пути на свои)
    name = os.path.basename(inp_file).split('.')[0]
    return f'C:\\Users\\user\\Intelligent_system_generating_graphic_objects\\output_instantmesh\\instant-mesh-large\\meshes\\{name}.obj'

#app 1
model_page =  gr.Interface( fn=reconstruction_object,
    inputs=gr.Image(type='filepath'), title="Интеллектуальная система генерации и анализа графических объектов \n SergioGreenDragonGenerate",
    outputs=gr.Model3D(clear_color=(0.0, 0.0, 0.0, 0.0), camera_position=(0.0, 0.0, 0.0), zoom_speed=0.8,
    pan_speed=0.7, label="3D Model", interactive=True))


with gr.Blocks(theme=seafoam) as demo:
    # inp_file - путь до картинки
    with gr.Tab("Построить трёхмерную модель"):
        model_page = gr.Interface(fn=reconstruction_object,
                                inputs=gr.Image(type='filepath'),
                                title="Интеллектуальная система генерации и анализа графических объектов \n SergioGreenDragonGenerate",
                                outputs=gr.Model3D(clear_color=(0.0, 0.0, 0.0, 0.0),
                                zoom_speed=0.8,
                                pan_speed=0.7, label="3D Model"))

    with gr.Tab("Модель быстрой реконструкции TripoSR") as page_tripo:
        with gr.Blocks():
            gr.Markdown(
                """
            # TripoSR Demo
            [TripoSR](https://github.com/VAST-AI-Research/TripoSR) is a state-of-the-art open-source model for **fast** feedforward 3D reconstruction from a single image, collaboratively developed by [Tripo AI](https://www.tripo3d.ai/) and [Stability AI](https://stability.ai/).

            **Tips:**
            1. If you find the result is unsatisfied, please try to change the foreground ratio. It might improve the results.
            2. It's better to disable "Remove Background" for the provided examples (except fot the last one) since they have been already preprocessed.
            3. Otherwise, please disable "Remove Background" option only if your input image is RGBA with transparent background, image contents are centered and occupy more than 70% of image width or height.
            """
            )
            with gr.Row(variant="panel"):
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(
                            label="Input Image",
                            image_mode="RGBA",
                            sources="upload",
                            type="pil",
                            elem_id="content_image",
                        )
                        processed_image = gr.Image(label="Processed Image", interactive=False)
                    with gr.Row():
                        with gr.Group():
                            do_remove_background = gr.Checkbox(
                                label="Remove Background", value=True
                            )
                            foreground_ratio = gr.Slider(
                                label="Foreground Ratio",
                                minimum=0.5,
                                maximum=1.0,
                                value=0.85,
                                step=0.05,
                            )
                            mc_resolution = gr.Slider(
                                label="Marching Cubes Resolution",
                                minimum=32,
                                maximum=320,
                                value=256,
                                step=32
                            )
                    with gr.Row():
                        submit = gr.Button("Generate", elem_id="generate_1", variant="primary")
                with gr.Column():
                    with gr.Tab("OBJ"):
                        output_model_obj = gr.Model3D(
                            label="Output Model (OBJ Format)",
                            interactive=False,
                        )
                        gr.Markdown("Note: The model shown here is flipped. Download to get correct results.")
                    with gr.Tab("GLB"):
                        output_model_glb = gr.Model3D(
                            label="Output Model (GLB Format)",
                            interactive=False,
                        )
                        gr.Markdown(
                            "Note: The model shown here has a darker appearance. Download to get correct results.")
            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[
                        "examples/hamburger.png",
                        "examples/poly_fox.png",
                        "examples/robot.png",
                        "examples/teapot.png",
                        "examples/tiger_girl.png",
                        "examples/horse.png",
                        "examples/flamingo.png",
                        "examples/unicorn.png",
                        "examples/chair.png",
                        "examples/iso_house.png",
                        "examples/marble.png",
                        "examples/police_woman.png",
                        "examples/captured.jpeg",
                    ],
                    inputs=[input_image],
                    outputs=[processed_image, output_model_obj, output_model_glb],
                    cache_examples=False,
                    fn=partial(run_example),
                    label="Examples",
                    examples_per_page=20,
                )
            submit.click(fn=check_input_image, inputs=[input_image]).success(
                fn=preprocess,
                inputs=[input_image, do_remove_background, foreground_ratio],
                outputs=[processed_image],
            ).success(
                fn=generate,
                inputs=[processed_image, mc_resolution],
                outputs=[output_model_obj, output_model_glb],
            )

    with gr.Tab("Модель эффективной реконструкции InstantMesh") as page_instantmesh:

        ###############################################################################
        # Configuration.
        ###############################################################################

        # Define the cache directory for model files
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            device0 = torch.device('cuda:0')
            device1 = torch.device('cuda:1')
        else:
            device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device1 = device0

        # Define the cache directory for model files
        model_cache_dir = './ckpts/'
        os.makedirs(model_cache_dir, exist_ok=True)

        seed_everything(0)

        config_path = 'configs/instant-mesh-large.yaml'
        config = OmegaConf.load(config_path)
        config_name = os.path.basename(config_path).replace('.yaml', '')
        model_config = config.model_config
        infer_config = config.infer_config

        IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

        device = torch.device('cuda')

        # load diffusion model
        print('Loading diffusion model ...')
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
            cache_dir=model_cache_dir
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )

        # load custom white-background UNet
        unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin",
                                         repo_type="model", cache_dir=model_cache_dir)
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        pipeline.unet.load_state_dict(state_dict, strict=True)

        pipeline = pipeline.to(device0)

        # load reconstruction model
        print('Loading reconstruction model ...')
        model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt",
                                          repo_type="model", cache_dir=model_cache_dir)
        model = instantiate_from_config(model_config)
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if
                      k.startswith('lrm_generator.') and 'source_camera' not in k}
        model.load_state_dict(state_dict, strict=True)

        model = model.to(device1)
        if IS_FLEXICUBES:
            model.init_flexicubes_geometry(device1, fovy=30.0)
        model = model.eval()

        print('Loading Finished!')

        with gr.Blocks():
            gr.Markdown(_HEADER_)
            with gr.Row(variant="panel"):
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(
                            label="Input Image",
                            image_mode="RGBA",
                            sources="upload",
                            width=256,
                            height=256,
                            type="pil",
                            elem_id="content_image",
                        )
                        processed_image = gr.Image(
                            label="Processed Image",
                            image_mode="RGBA",
                            width=256,
                            height=256,
                            type="pil",
                            interactive=False
                        )
                    with gr.Row():
                        with gr.Group():
                            do_remove_background = gr.Checkbox(
                                label="Remove Background", value=True
                            )
                            sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

                            sample_steps = gr.Slider(
                                label="Sample Steps",
                                minimum=30,
                                maximum=75,
                                value=75,
                                step=5
                            )

                    with gr.Row():
                        submit = gr.Button("Generate", elem_id="generate_2", variant="primary")

                    with gr.Row(variant="panel"):
                        gr.Examples(
                            examples=[
                                os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
                            ],
                            inputs=[input_image],
                            label="Examples",
                            examples_per_page=20
                        )

                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            mv_show_images = gr.Image(
                                label="Generated Multi-views",
                                type="pil",
                                width=379,
                                interactive=False
                            )

                        with gr.Column():
                            output_video = gr.Video(
                                label="video", format="mp4",
                                width=379,
                                autoplay=True,
                                interactive=False
                            )

                    with gr.Row():
                        with gr.Tab("OBJ"):
                            output_model_obj = gr.Model3D(
                                label="Output Model (OBJ Format)",
                                # width=768,
                                interactive=False,
                            )
                            gr.Markdown(
                                "Note: Downloaded .obj model will be flipped. Export .glb instead or manually flip it before usage.")
                        with gr.Tab("GLB"):
                            output_model_glb = gr.Model3D(
                                label="Output Model (GLB Format)",
                                # width=768,
                                interactive=False,
                            )
                            gr.Markdown(
                                "Note: The model shown here has a darker appearance. Download to get correct results.")

                    with gr.Row():
                        gr.Markdown(
                            '''Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).''')

            gr.Markdown(_CITE_)
            mv_images = gr.State()

            submit.click(fn=check_input_image, inputs=[input_image]).success(
                fn=preprocess_instantmesh,
                inputs=[input_image, do_remove_background],
                outputs=[processed_image],
            ).success(
                fn=generate_mvs,
                inputs=[processed_image, sample_steps, sample_seed],
                outputs=[mv_images, mv_show_images],
            ).success(
                fn=make3d,
                inputs=[mv_images],
                outputs=[output_video, output_model_obj, output_model_glb]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
