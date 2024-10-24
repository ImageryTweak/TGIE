
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import base64
import requests
import sys
from loguru import logger
from pathlib import Path

######################################
#       Step 4 Run Inpainting
######################################

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#import dashscope
from PIL import Image, ImageOps
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms as T
from diffusers.utils import load_image
from diffusers import DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
from mllm.dialoggen_demo import DialogGen,eval_model,init_dialoggen_model
from hydit.config import get_args
from hydit.inference_controlnet import End2End
#sys.path.insert(0,os.path.join(os.path.abspath('.'),"third_party/InfEdit"))
#from third_party.InfEdit.app_infedit import inference

data = pd.read_csv('translate.csv')
start=0 #0
num=8   #1000
categories = list(data['category'][start:num])#改成全部
edit_objects = list(data['object'][start:num])

image_paths = ['data_test/' + path for path in data['instruction_target_image']][start:num]
masks_paths = [f'data_test/mask_en/image_{i+1}.png' for i in range(len(categories))]
#masks_paths = [f'data_test/mask/image_{i+1}.png' for i in range(len(categories))]
editing_prompts = list(data['translate'][start:num])
result_images = []
results_path = "results-global"


norm_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )



def inferencer():

    args = get_args()
    models_root_path = Path(args.model_root)   #models_root_path = "ckpts"
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    args.control_type = 'canny'
    args.load_key='distill'

    #print("args=",args)
    gen = End2End(args, models_root_path)
    enhancer = None

    return args, gen, enhancer


args, gen, enhancer = inferencer()

# Run inference
logger.info("Generating images...")
height, width = args.image_size

models_captioner_path='/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/ckpts/dialoggen'
models_captioner = init_dialoggen_model(models_captioner_path)

for image_path_i in range(len(image_paths)):
    print("image_path_i为",image_path_i)
    enhanced_prompt = None
    image_path = image_paths[image_path_i]
    editing_prompt = editing_prompts[image_path_i]
    category = categories[image_path_i]
    edit_object = edit_objects[image_path_i]
    mask_path = masks_paths[image_path_i]

    # mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    # mask_np = mask / 255

    
    if category == "Global":
        
        print("category=",category)
        print("editing_prompt为",editing_prompt)
        query = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{editing_prompt}"
        res = eval_model(models_captioner,query=query,image_file=image_path)
        #print(res)
        height, width = args.image_size
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # 转换为灰度图像
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)   # Canny 边缘检测
        
        # 将单通道的灰度图像复制到三个通道，形成RGB图像
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        resized_edges = cv2.resize(edges_rgb, (width, height))# 调整图像大小
        # 等价于 图片.convert('RGB').resize((width, height)) #condition
        
        edge_image = norm_transform(resized_edges)
        edge_image = edge_image.unsqueeze(0).cuda()

        results = gen.predict(res,
                              height=height,
                              width=width,
                              image=edge_image,
                              seed=args.seed,
                              enhanced_prompt=enhanced_prompt,
                              negative_prompt=args.negative,
                              infer_steps=args.infer_steps,
                              guidance_scale=args.cfg_scale,
                              batch_size=args.batch_size,
                              src_size_cond=args.size_cond,
                              use_style_cond=args.use_style_cond,
                              )
        images = results['images']
        save_dir = Path(results_path)
        for idx, pil_img in enumerate(images):
            save_path = save_dir / f"{image_path_i+start}.png"
            pil_img.save(save_path)
            logger.info(f"Save to {save_path}")

    else:
        continue
        
    
