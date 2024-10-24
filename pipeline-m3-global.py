
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

import base64
import requests
import sys
from loguru import logger
from pathlib import Path
import csv
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

data = pd.read_csv('translate.csv')
New_Captions_data = pd.read_csv('mllm/New_Caption_all.csv')
start=1000 #0
num=1213   #1000
categories = list(data['category'][start:num])#改成全部
edit_objects = list(data['object'][start:num])
New_Captions = list(New_Captions_data['New Caption'][30:])

image_paths = ['data_test/' + path for path in data['instruction_target_image']][start:num]
masks_paths = [f'data_test/mask_en/image_{i+1}.png' for i in range(len(categories))]
#masks_paths = [f'data_test/mask/image_{i+1}.png' for i in range(len(categories))]
editing_prompts = list(data['translate'][start:num])
result_images = []
results_path = "results-global-rest"
image_paths_indices = []

norm_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

def task_to_prompt(control_type):
    if control_type == "object-removal":
        promptA = "P_ctxt"
        promptB = "P_ctxt"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"
    elif control_type == "context-aware":
        promptA = "P_ctxt"
        promptB = "P_ctxt"
        negative_promptA = ""
        negative_promptB = ""
    elif control_type == "shape-guided":
        promptA = "P_shape"
        promptB = "P_ctxt"
        negative_promptA = "P_shape"
        negative_promptB = "P_ctxt"
    elif control_type == "image-outpainting":
        promptA = "P_ctxt"
        promptB = "P_ctxt"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"
    else:
        promptA = "P_obj"
        promptB = "P_obj"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


def inferencer():
    
    args = get_args()
    models_root_path = Path(args.model_root)   #models_root_path = "ckpts"
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    
    args.control_type = 'canny'
    args.load_key='distill'
    
    print("args=",args)
    gen = End2End(args, models_root_path)

    # # Try to enhance prompt
    # if args.enhance:
    #     logger.info("Loading DialogGen model (for prompt enhancement)...")
    #     enhancer = DialogGen(str(models_root_path / "dialoggen"), args.load_4bit)
    #     logger.info("DialogGen model loaded.")
    # else:
    #     enhancer = None
    enhancer = None
    
    return args, gen, enhancer


args, gen, enhancer = inferencer()


# Run inference
logger.info("Generating images...")
height, width = args.image_size

models_captioner_path='/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/ckpts/dialoggen'
#models_captioner = init_dialoggen_model(models_captioner_path)

t=0 #指示mllm/New_Caption.csv
csv_file_name = "title_save.csv"
for image_path_i in range(start,len(image_paths)+start):
    print("image_path_i为",image_path_i)
    enhanced_prompt = None
    image_path = image_paths[image_path_i-start]
    editing_prompt = editing_prompts[image_path_i-start]
    category = categories[image_path_i-start]
    edit_object = edit_objects[image_path_i-start]
    mask_path = masks_paths[image_path_i-start]


    
    if category == "Global":
        
        print("category=",category)
        print("editing_prompt为",editing_prompt)
        query = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{editing_prompt}"
        #res = eval_model(models_captioner,query=query,image_file=image_path)
        #print(res)
        print("image_path_i=",image_path_i)
        #print("start=",start)
        res = New_Captions[t]
        print(res)
        t = t+1
        #res = "<画图>一位女士站在木制码头上，手里握着红色的雨伞，迎着阳光，背景是平静的湖面和远处的山脉，镜头是全景，风格是水彩风格。"
        
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
            save_path = save_dir / f"{image_path_i}.png"
            pil_img.save(save_path)
            logger.info(f"Save to {save_path}")
        image_paths_indices.append(image_path_i)
        
        with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入标题行
            writer.writerow(['results-path'])
            
            # 遍历列表，写入每一行数据
            writer.writerow([f"results-global-new/{image_path_i}.png"])


    else:
        continue
        
