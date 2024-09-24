import os
import base64
import requests
import sys
import pandas as pd


################################################################
#                      Config
################################################################
# enter your qianwen api key here
qianwen_key = "sk-aa1a35a2d0454a8b8ed7e183633f2989"
#  base 64 编码格式
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
# define image_path and prompt
total_image_path = 'data/challenge_dataset/'
img_mask_path = 'data/challenge_dataset/mask'
prompt_file = pd.read_csv(total_image_path + '/translate_sheet1.csv')
image_names = list(prompt_file['instruction_target_image'].str[22:])
prompt_ch = list(prompt_file['translate'])


################################################################
#         Step 1 Confirm Editing Category and Object
################################################################

def get_response_type(image_path, editing_prompt, qianwen_key):
    base64_image = encode_image(image_path)
    api_key = qianwen_key
    headers = {
       "Content-Type": "application/json",
       "Authorization": f"Bearer {api_key}"
       }
    payload = {
        "model": "qwen-vl-max",
        "messages": [
            {
                "role": "user",
                "content": [


                    {
                        "type": "text",
                        "text": "我会给你一张图片和一条编辑指令，请输出它属于哪种编辑类别，你必须从以下类别中选择: \n\
                        1. 增加：在图像中增加新的对象。\n\
                        2. 删除: 删除图像中的某个对象或某些对像 \n\
                        3. 局部：替换对象的局部部分并随后改变对象的属性（例如，让它微笑），或在不影响对象结构的情况下改变其视觉外观（例如，将猫变成狗）。 \n\
                        4. 全局：编辑整个图像的风格，例如，将图像呈现为冬季景象。 \n\
                        5. 背景：更改场景的背景，例如，让他在草原上。"
                    },
                    {
                        "type": "text",
                        "text": editing_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],"max_tokens": 300
        }
    response = requests.post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", headers=headers, json=payload)
    response_str = response.json()["choices"][0]["message"]["content"]
    # print(response_str)
    return response_str

def get_response_object(image_path, editing_prompt, qianwen_key):
    base64_image = encode_image(image_path)
    api_key = qianwen_key
    headers = {
       "Content-Type": "application/json",
       "Authorization": f"Bearer {api_key}"
       }
    payload = {
        "model": "qwen-vl-plus",
        "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "我会给你图像的编辑指令，请根据指令输出需要编辑的对象，用不超过五个字来描述对象，不要输出其他内容。描述应仅包含来自图像的信息。不要添加编辑指令的信息。输出应只包含一个名词。"
                            }, ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": editing_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
        }
    response = requests.post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", headers=headers, json=payload)
    response_str = response.json()["choices"][0]["message"]["content"]
    return response_str


categories = []
edit_objects = []
for image_path_i in range(len(image_names)):
    image_path = total_image_path + image_names[image_path_i]
    editing_prompt = prompt_ch[image_path_i]
    re_category = get_response_type(image_path,editing_prompt,qianwen_key)
    # print(f"image path: {image_path}")
    print(f"editing prompt: {editing_prompt}")
    for category_name in ["Addition","Remove","Local","Global","Background"]:
        if category_name in re_category:
            categories.append(category_name)
            break
    category = categories[image_path_i]
    print(f"category: {category}")
    if category in ["Background", "Global", "Addition"]:
        edit_objects.append("nan")
        print(f"object: {edit_objects[image_path_i]}")
        continue
    res_obj = get_response_object(image_path,editing_prompt,qianwen_key)
    edit_objects.append(res_obj)
    print(f"object: {edit_objects[image_path_i]}")



##############################################################
#              Step 2&3   Segmentation&mask
##############################################################
sys.path.insert(0, os.path.join(os.path.abspath('.'), 'third_party/GSAM'))
from PIL import Image
import numpy as np
from third_party.GSAM.app import run_grounded_sam
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import torch

with open('object_ch2en.txt', 'r',encoding='utf-8') as file:
    text1 = file.read()
# 根据 'image path' 进行分割
object_ch = text1.split('\n')

with open('category.txt', 'r',encoding='utf-8') as file:
    text3 = file.read()
# 根据 'image path' 进行分割
category_ch = text3.split('\n')


birefnet = AutoModelForImageSegmentation.from_pretrained('/home2/lsy/HunyuanDiT_No.3/BiRefNet', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()
def extract_object(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask



masks = []
for image_path_i in range(len(image_names)):
    image_path = image_names[image_path_i]
    editing_prompt = prompt_ch[image_path_i]
    category = category_ch[image_path_i]
    edit_object = object_ch[image_path_i]

    if category == "Addition":
        while True:
            base64_image = encode_image(image_path)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {qianwen_key}"
            }
            payload = {
                "model": "qwen-vl-plus",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "我需要根据指令 " + editing_prompt + " + 将一个对象添加到图像中。图像的大小为 640（高度和宽度均为 640 像素）。左上角坐标为 [0, 0]，右下角坐标为 [640, 640]。请给出添加对象的位置的可能边界框。请按以下格式输出：[左上角 x 坐标，左上角 y 坐标，框宽度，框高度]。你只需要输出边界框的位置，不需要其他内容。请参考下面的示例格式。\n\
                                [19, 101, 32, 153]\n\
                                [54, 12, 242, 96]"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }

            response = requests.post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", headers=headers, json=payload)

            response_str = response.json()["choices"][0]["message"]["content"]

            try:
                box = response_str[1:-1].split(",")
                for i in range(len(box)):
                    box[i] = int(box[i])

                cus_mask = np.zeros((640, 640))
                cus_mask[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = 255

                masks.append(cus_mask)
                print(f"image path: {image_path}")
                print(f"editing prompt: {editing_prompt}")
                print(f"category: {category}")
                print(f"response: {response_str}")
                break
            except:
                continue
        continue
    elif category == "Background":
        _, mask_c = extract_object(birefnet, imagepath=image_path)
        masks.append(mask_c)
    elif category == "Global":
        masks.append(255 * np.ones((640, 640)))
        # print(f"image path: {image_path}")
        # print(f"editing prompt: {editing_prompt}")
        # print(f"category: {category}")
        continue
    else:
        labels = edit_object

    for thresh in [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]:
        try:
            detections = run_grounded_sam(
                input_image={"image": Image.open(image_path), "mask": None},
                text_prompt=labels,
                task_type="seg",
                inpaint_prompt="",
                box_threshold=thresh,
                text_threshold=0.25,
                iou_threshold=0.5,
                inpaint_mode="merge",
                scribble_mode="split"
            )
            masks.append(np.array(detections[0, 0, ...].cpu()) * 255)
            print(f"image path: {image_path}")
            print(f"editing prompt: {editing_prompt}")
            print(f"category: {category}")
            break
        except:
            print(f"wrong in threshhold: {thresh}")


for image_path_i in range(len(image_names)):
    image_path = image_names[image_path_i]
    editing_prompt = prompt_ch[image_path_i]
    category = category_ch[image_path_i]
    edit_object = object_ch[image_path_i]
    mask = masks[image_path_i]
    print(f"image path: {image_path}")
    print(f"editing prompt: {editing_prompt}")
    print(f"category: {category}")
    print(f"edit object: {edit_object}")
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.save(img_mask_path+image_path)
