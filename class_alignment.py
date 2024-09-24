import pandas as pd
import clip
import torch
from tqdm import tqdm
import json
import numpy as np
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

def combine_class(class_name):
    all_class_dict={}
    for i in range(len(class_name)):
        path=f'./data/{class_name[i]}_class.csv'
        current_class_dict=pd.read_csv(path).to_dict(orient='records')
        print(current_class_dict)
        for j in range(len(current_class_dict)):
            current_class_name=current_class_dict[j]['class'].lower()
            if current_class_name not in all_class_dict:
                all_class_dict[current_class_name]=str()
            else:
                print(current_class_name)
    all_class_list=[]
    for key in all_class_dict:
        all_class_list.append({'class':key})
    all_class_list_df=pd.DataFrame(all_class_list)
    all_class_list_df.to_csv('./data/groundingDINO_class.csv',index=False)



def process_embedding():
    path = './data/groundingDINO_class.csv'
    class_name_dict = pd.read_csv(path).to_dict(orient='records')
    result_dict=[]
    for i in tqdm(range(len(class_name_dict))):
        current_class_name=class_name_dict[i]['class']
        text = clip.tokenize([current_class_name]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text).cpu().numpy().tolist()
            text_features = '{' + str(text_features[0])[1:-1] + '}'
            result_dict.append({'id':i+1,'class':current_class_name,'text_features':text_features})
    if result_dict != []:
        result_dict_df = pd.DataFrame(result_dict)
        result_dict_df.to_csv('./data/groundingDINO_class_embedding.csv', index=False)
    return model


def process_similarity(object_list, embedding_csv_path, top_n=5):
    # 读取 CSV 文件
    df = pd.read_csv(embedding_csv_path)
    df['text_features'] = df['text_features'].apply(lambda x: np.fromstring(x.strip('{}'), sep=','))
    # result_list 用于保存输出结果
    result_list = []

    for i in tqdm(range(len(object_list))):
        text = clip.tokenize(object_list[i]).to(device)  # 将文本编码为 tensor
        with torch.no_grad():
            text_features = model.encode_text(text).cpu().numpy().tolist()[0]
            print(text_features)
            result = recall(text_features, df)
            result_list.append(result)
    return result_list



def recall(embedding, df, top_n=5):
    # 计算余弦相似度
    similarity_scores = []
    for index, row in df.iterrows():
        emb = row['text_features']

        if np.linalg.norm(emb) == 0 or np.linalg.norm(embedding) == 0:
            similarity = 0  # 避免除以零
        else:
            similarity = np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))

        similarity_scores.append((row, similarity))

    # 按照相似度排序，并取前 N 个结果
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in similarity_scores[:top_n]]


if __name__ == '__main__':

    # #read&combine all class, as long as you have groundingDINO_class.csv, you don't need to run this part
    # class_name=['coco','o365','openimage']
    # combine_class(class_name)

    # # compute embedding, as long as you have groundingDINO_class_embedding.csv, you don't need to run this part
    # embedding_info = process_embedding()

    # read_valid_class_ch2en
    with open('object_ch2en.txt', 'r', encoding='utf-8') as file:
        process_data = file.read()
    # 根据 'image path' 进行分割
    object_ch = process_data.split('\n')
    result_list = process_similarity(object_ch,'./data/groundingDINO_class_embedding.csv')
    with open('object_align.txt', 'w') as file:
        for string in result_list:
            file.write(string + "\n")