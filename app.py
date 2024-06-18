#define your first application
#!pip install gradio
import gradio as gr
import os
import torch
#import seaborn as sns
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

# inp_file - путь до картинки

css1 = '''
.main_foreground{
    background-image: url("C:/Users/user/Intelligent_system_generating_graphic_objects/output/0/input.png");
}
'''
theme = gr.themes.Glass(
    primary_hue="emerald",
    text_size="md",
    font=['Bahnschrift', 'Candara', 'Noto Sans', 'source-sans-pro'],
).set(
    body_background_fill='*primary_500',
    body_text_color='*neutral_900',
    body_text_weight='500',
    background_fill_primary='*neutral_100',
    shadow_drop='*shadow_drop_lg',
    button_border_width='3px'
)

def reconstruction_object(inp_file):
    new_obj=''
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    size_work_memory = torch.cuda.mem_get_info()[0]
    #print("CUDA memory_reserved:" + str(r))
    #print("CUDA memory_allocated:" + str(a))
    if torch.cuda.is_available() and size_work_memory>=12884901888:
        new_obj=run_instantmesh(inp_file)
    else:
        new_obj = run_tripo(inp_file)

    return new_obj

def fn1():
    return 0
def run_tripo(inp_file):
    # запуск скрипта run.py
    os.system(f'python tripo_run.py {inp_file} --output-dir output_tripo/')
    # вернуть путь до 3D результата (заменить пути на свои)
    return 'C:\\Users\\user\\Intelligent_system_generating_graphic_objects\\output_tripo\\0\\mesh.obj'

def run_instantmesh(inp_file):
    # запуск скрипта run.py
    os.system(f'python Instantmesh_run.py configs/instant-mesh-large.yaml {inp_file} --output_path output_instantmesh --save_video')
    # вернуть путь до 3D результата (заменить пути на свои)
    return 'C:\\Users\\user\\Intelligent_system_generating_graphic_objects\\output_instantmesh\\instant-mesh-large\\meshes\\image.obj'

#app 1
model_page =  gr.Interface( fn=reconstruction_object,
    inputs=gr.Image(type='filepath'),
    outputs=gr.Model3D(clear_color=(0.0, 0.0, 0.0, 0.0), label="3D Model", interactive=False,), css=css1)

inputs = [
    ]
outputs = gr.Plot()

#app 2
app2 = gr.Interface(fn=fn1, inputs=inputs, outputs=outputs)
#combine to create a multipage app
demo = gr.TabbedInterface([model_page, app2], ["Построить трёхмерную модель", "Name2"],
                          title="Интеллектуальная система генерации и анализа графических объектов \n SergioGreenDragonGenerate", theme=theme,
                             css=css1                                                                  )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
