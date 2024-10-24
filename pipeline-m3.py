import sys
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import dashscope
from torchvision import transforms as T
from hydit.config import get_args
from hydit.inference_controlnet import End2End
from loguru import logger
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageEnhance
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils import load_image
from diffusers import DPMSolverMultistepScheduler
from mllm.dialoggen_demo import DialogGen,eval_model,init_dialoggen_model
from hydit.config import get_args
from hydit.inference_controlnet import End2End
import base64
import requests

api_key = 'input your api key'

norm_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )
#inferencer是局部编辑，inferencer2是全局编辑，（若DIT Model一致）两者可以合并
def inferencer():
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    # Load models
    gen = End2End(args, models_root_path)
    # Try to enhance prompt
    if args.enhance:
        logger.info("Loading DialogGen model (for prompt enhancement)...")
        enhancer = DialogGen(str(models_root_path / "dialoggen"), args.load_4bit)
        logger.info("DialogGen model loaded.")
    else:
        enhancer = None
    return args, gen, enhancer

def inferencer2():
    args2 = get_args()
    models_root_path = Path(args2.model_root)  # models_root_path = "ckpts"
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    args2.control_type = 'canny'
    args2.load_key = 'distill'
    gen2 = End2End(args2, models_root_path)
    enhancer2 = None
    return args2, gen2, enhancer2

args, gen, enhancer = inferencer()
args2, gen2, enhancer2 = inferencer2()

if enhancer:
    logger.info("Prompt Enhancement...")
    success, enhanced_prompt = enhancer(args.prompt)
    if not success:
        logger.info("Sorry, the prompt is not compliant, refuse to draw.")
        exit()
    logger.info(f"Enhanced prompt: {enhanced_prompt}")
else:
    enhanced_prompt = None

# Run inference
logger.info("Generating images...")
height, width = args.image_size

data = pd.read_csv('translate.csv')
start=0 #start_id
num=8   #end_id
categories = list(data['category'][start:num])#改成全部
edit_objects = list(data['object'][start:num])
image_paths = ['data_test/' + path for path in data['instruction_target_image']][start:num]
masks_paths = [f'data_test/mask_en/image_{i+1}.png' for i in range(len(categories))]
editing_prompts = list(data['translate'][start:num])
result_images = []
results_path = "results"

models_captioner_path='/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/ckpts/dialoggen'
models_captioner = init_dialoggen_model(models_captioner_path)

for image_path_i in range(len(image_paths)):
    enhanced_prompt = None
    image_path = image_paths[image_path_i]
    editing_prompt = editing_prompts[image_path_i]
    category = categories[image_path_i]
    edit_object = edit_objects[image_path_i]
    mask_path = masks_paths[image_path_i]
    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(mask)
        continue
    mask_np = mask / 255
    messages = [
        {
            "role": "system",
            "content": [
                {
                'type': 'text',
                'text': '我将会给你一段图像的修改介绍，请输出修改后的图像应该存在的新内容。例如：图像修改介绍为”把白猫修改为黑狗“，你应该输出“黑狗”，只输出新内容，不要输出其它内容。强调：只输出新内容，不要输出其它内容'
                },]
         },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': editing_prompt
                }
            ]
        }
    ]
    response = dashscope.Generation.call(
        'qwen2-72b-instruct',
        api_key=api_key,
        messages=messages,
        seed=1234,  # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message"  format.
        stream=False,
    )
    if response.status_code == 200:
        response_str = response.output.choices[0]['message']['content']
        print(response_str)
    else:
        print(response.status_code)
        continue
    if category == "Local":
        #add mask to image
        image_masked = cv2.imread(image_path)[:, :, ::-1]
        for i in range(image_masked.shape[0]):
            for j in range(image_masked.shape[1]):
                if mask_np[i, j] == 1.0:
                    image_masked[i, j] = 0
        mask_np = mask_np[:, :, np.newaxis]
        caption = response_str
        height, width = args.image_size
        image_masked = Image.fromarray(image_masked).resize((width, height)).convert('RGB')
        image_masked = norm_transform(image_masked)
        image_masked = image_masked.unsqueeze(0).cuda(device=2)
        results = gen.predict(caption,
                              height=height,
                              width=width,
                              image=image_masked,
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
        # enhancer = ImageEnhance.Brightness(image)
        # image = enhancer.enhance(1.2)

        # image_np = np.array(image)
        # init_image_np = cv2.imread(image_path)[:, :, ::-1]
        #
        # image_pasted = init_image_np * (1 - mask_blurred) + mask_blurred * image_np
        # image_pasted = image_pasted.astype(image_np.dtype)
        #image = Image.fromarray(image_pasted)
        #image = Image.fromarray(image.astype(np.uint8))
        # image.save(f"results/{image_path_i}.jpg")
        # result_images.append(image)

    elif category == "Background":
        image_masked = cv2.imread(image_path)[:, :, ::-1]
        mask_np = mask_np[:, :, np.newaxis]
        if not (mask_np[94:547,94:546].sum() < mask_np.sum() - mask_np[94:547,94:546].sum() and mask_np[0,:].sum()>0 and mask_np[-1,:].sum()>0 and mask_np[:,0].sum()>0 and mask_np[:,-1].sum()>0)   and mask_np[1,:].sum()>0 and mask_np[-2,:].sum()>0 and mask_np[:,1].sum()>0 and mask_np[:,-2].sum()>0 :
            mask_np=1-mask_np
        for i in range(image_masked.shape[0]):
            for j in range(image_masked.shape[1]):
                if mask_np[i, j] == 0.0:
                    image_masked[i, j] = 0
        height, width = args.image_size
        image_masked = Image.fromarray(image_masked).resize((width, height)).convert('RGB')
        image_masked = norm_transform(image_masked)
        image_masked = image_masked.unsqueeze(0).cuda(device=2)
        mask_np = mask_np[:, :, np.newaxis]

        caption = response_str
        height, width = args.image_size
        results = gen.predict(caption,
                              height=height,
                              width=width,
                              image=image_masked,
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


    elif category == "Global":
        query = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{editing_prompt}"
        res = eval_model(models_captioner, query=query, image_file=image_path)
        print(res)
        height, width = args2.image_size
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)  # Canny 边缘检测

        # 将单通道的灰度图像复制到三个通道，形成RGB图像
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        resized_edges = cv2.resize(edges_rgb, (width, height))  # 调整图像大小
        # 等价于 图片.convert('RGB').resize((width, height)) #condition

        edge_image = norm_transform(resized_edges)
        edge_image = edge_image.unsqueeze(0).cuda()
        results = gen2.predict(res,
                              height=height,
                              width=width,
                              image=edge_image,
                              seed=args2.seed,
                              enhanced_prompt=enhanced_prompt,
                              negative_prompt=args2.negative,
                              infer_steps=args2.infer_steps,
                              guidance_scale=args2.cfg_scale,
                              batch_size=args2.batch_size,
                              src_size_cond=args2.size_cond,
                              use_style_cond=args2.use_style_cond,
                              )
        images = results['images']
        save_dir = Path(results_path)
        for idx, pil_img in enumerate(images):
            save_path = save_dir / f"{image_path_i + start}.png"
            pil_img.save(save_path)
            logger.info(f"Save to {save_path}")

    elif category == "Remove":
        #add mask to image
        image_masked = cv2.imread(image_path)[:, :, ::-1]
        for i in range(image_masked.shape[0]):
            for j in range(image_masked.shape[1]):
                if mask_np[i, j] == 1.0:
                    image_masked[i, j] = 0

        caption = response_str
        height, width = args.image_size
        image_masked = Image.fromarray(image_masked).resize((width, height)).convert('RGB')
        image_masked = norm_transform(image_masked)
        image_masked = image_masked.unsqueeze(0).cuda(device=2)
        results = gen.predict("",
                              height=height,
                              width=width,
                              image=image_masked,
                              seed=args.seed,
                              enhanced_prompt=enhanced_prompt,
                              negative_prompt=edit_object + "超出框架, 低分辨率,错误,裁剪,最差质量,低质量，JPEG伪影，丑陋，重复，病态，残缺，变异，变形，模糊，脱水，解剖不良，比例失调，多余的四肢，畸形，粗糙的比例，多余的肢体，水印，签名,糟糕的艺术,毁容,错误的眼睛，糟糕的人脸",
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


    elif category == "Addition":
        image_masked = cv2.imread(image_path)[:, :, ::-1]
        for i in range(image_masked.shape[0]):
            for j in range(image_masked.shape[1]):
                if mask_np[i, j] == 1.0:
                    image_masked[i, j] = 0
        image_masked.save(f"results/f{image_path_i+start}.jpg")
        caption = response_str
        height, width = args.image_size
        image_masked = Image.fromarray(image_masked).resize((width, height)).convert('RGB')
        image_masked = norm_transform(image_masked)
        image_masked = image_masked.unsqueeze(0).cuda(device=2)
        results = gen.predict(caption,
                              height=height,
                              width=width,
                              image=image_masked,
                              seed=args.seed,
                              enhanced_prompt=enhanced_prompt,
                              negative_prompt="",
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

