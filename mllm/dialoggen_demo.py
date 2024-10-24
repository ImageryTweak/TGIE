import argparse
import torch
import sys
import os
# 添加当前命令行运行的目录到 sys.path
sys.path.append(os.getcwd()+"/mllm")
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from PIL import Image
from io import BytesIO
import re

import pandas as pd
import csv

def image_parser(image_file, sep=','):
    out = image_file.split(sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def init_dialoggen_model(model_path, model_base=None, load_4bit=False):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, llava_type_model=True, load_4bit=load_4bit)
    return {"tokenizer": tokenizer,
            "model": model,
            "image_processor": image_processor}


def eval_model(models,
               query='详细描述一下这张图片',
               image_file=None,
               sep=',',
               temperature=0.2,
               top_p=None,
               num_beams=1,
               max_new_tokens=512,
               return_history=False,
               history=None,
               skip_special=False
               ):
    # Model
    disable_torch_init()

    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if models["model"].config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if models["model"].config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            
    if not history:
        conv = conv_templates['llava_v1'].copy()
    else:
        conv = history

    if skip_special:
        conv.append_message(conv.roles[0], query)
    else:
        conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if image_file is not None:
        image_files = image_parser(image_file, sep=sep)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            models["image_processor"],
            models["model"].config
        ).to(models["model"].device, dtype=torch.float16)
    else:
        # fomatted input as training data
        image_sizes = [(1024, 1024)]
        images_tensor = torch.zeros(1, 5, 3, models["image_processor"].crop_size["height"], models["image_processor"].crop_size["width"])
        images_tensor = images_tensor.to(models["model"].device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, models["tokenizer"], IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    with torch.inference_mode():
        output_ids = models["model"].generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = models["tokenizer"].batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if return_history:
        return outputs, conv
    return outputs


def remove_prefix(text):
    if text.startswith("<画图>"):
        return text[len("<画图>"):], True
    elif text.startswith("对不起"):
        # 拒绝画图
        return "", False
    else:
        return text, True


class DialogGen(object):
    def __init__(self, model_path, load_4bit=False):
        self.models = init_dialoggen_model(model_path, load_4bit=load_4bit)
        self.query_template = "请先判断用户的意图，若为画图则在输出前加入<画图>:{}"

    def __call__(self, prompt, return_history=False, history=None, skip_special=False):
        enhanced_prompt = eval_model(
            models=self.models,
            query=self.query_template.format(prompt),
            image_file=None,
            return_history=return_history,
            history=history,
            skip_special=skip_special
        )
        if return_history:
            return enhanced_prompt

        enhanced_prompt, compliance = remove_prefix(enhanced_prompt)
        if not compliance:
            return False, ""
        return True, enhanced_prompt


if __name__ == "__main__":
    
    data = pd.read_csv('/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/translate.csv')
    start=0 #0
    num=1213   #1000
    #categories = list(data['category'][start:num])#改成全部
    
    #image_paths = ['/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/data_test/' + path for path in data['instruction_target_image']][start:num]
    image_paths = ['/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/data_test/image_edit_magic_data/image_680.png']
    
    #editing_prompts = list(data['translate'][start:num])
    # result_images = []
    # results_path = "results-global"
    
    #csv_file_name = "New_Caption_all.csv"
    results_caption = []  # 用于存储结果的列表
    image_paths_indices = []  # 用于存储图像路径索引的列表


    #parser = argparse.ArgumentParser()
    #parser.add_argument('--model_path', type=str, default='./ckpts/dialoggen')
    #parser.add_argument('--model_path', type=str, default='/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/ckpts/dialoggen')#dialoggen
    #parser.add_argument('--image_file', type=str, default='images/demo2.png') # 'images/demo1.jpeg'
    #parser.add_argument('--prompt', type=str, default='加一条鱼')
    #args = parser.parse_args()
    
    model_path='/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/ckpts/dialoggen'
    models = init_dialoggen_model(model_path)

    for image_path_i in range(len(image_paths)):
        image_path = image_paths[image_path_i]
        # editing_prompt = editing_prompts[image_path_i]
        # category = categories[image_path_i]
        
        '''if category == "Global":
            print("image_path_i为",image_path_i)
            print("editing_prompt为",editing_prompt)
            query = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{editing_prompt}"
            res = eval_model(models,
                query=query,
                image_file=image_path,
            )
            print(res)
            results_caption.append(res)
            image_paths_indices.append(image_path_i)
        else:
            continue'''

        print("image_path_i为",image_path_i)
        query1 = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{'雨天变成晴天'}"#editing_prompt
        query2 = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{'天气变成晴天'}"
        query3 = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{'变成阳光明媚'}"
        res1 = eval_model(models,
            query=query1,
            image_file=image_path,
        )
        res2 = eval_model(models,
            query=query2,
            image_file=image_path,
        )
        res3 = eval_model(models,
            query=query3,
            image_file=image_path,
        )
        print(res1)
        print(res2)
        print(res3)
        # results_caption.append(res)
        # image_paths_indices.append(image_path_i)

    # with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     # 写入标题行
    #     writer.writerow(['Image Path Index', 'New Caption'])
        
    #     # 遍历列表，写入每一行数据
    #     for index, result in zip(image_paths_indices, results_caption):
    #         writer.writerow([index, result])

###变成彩色
    # image_paths = ['/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/data_test/image_edit_magic_data/image_51.png',
    #           '/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/data_test/image_edit_magic_data/image_776.png',
    #           '/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/data_test/image_edit_magic_data/image_958.png']
    # editing_prompt = '图片变成彩色'
    # model_path='/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/ckpts/dialoggen'
    # models = init_dialoggen_model(model_path)
    # for image_path_i in range(len(image_paths)):
    #     image_path = image_paths[image_path_i]
    #     print("image_path_i为",image_path_i)

    #     query = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{editing_prompt}"
    #     res = eval_model(models,
    #         query=query,
    #         image_file=image_path,
    #     )
    #     print(res)
    
###后面要改prompt单独测的
    # image_paths = ['/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/data_test/image_edit_magic_data/image_51.png',
    #           '/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/data_test/image_edit_magic_data/image_776.png',
    #           '/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/data_test/image_edit_magic_data/image_958.png']
    # editing_prompt = '图片变成彩色'
    # model_path='/home/workspace/yuanyunhu/aigc/HunyuanDiT-main/ckpts/dialoggen'
    # models = init_dialoggen_model(model_path)
    # for image_path_i in range(len(image_paths)):
    #     image_path = image_paths[image_path_i]
    #     print("image_path_i为",image_path_i)

    #     query = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{editing_prompt}"
    #     res = eval_model(models,
    #         query=query,
    #         image_file=image_path,
    #     )
    #     print(res)
