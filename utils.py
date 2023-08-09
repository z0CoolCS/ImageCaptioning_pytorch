import os
from textwrap import wrap
import matplotlib.pyplot as plt
from PIL import Image
import json
from torchvision import transforms
from torchtext.vocab import vocab
from collections import Counter
import pandas as pd

def get_imageid(filename):
    id_image = filename.split('_')[-1].split('.')[0]
    return id_image.lstrip('0')

def get_annot_by_imageid(data_json, unique=False):
    annot_by_imageid = {}
    if unique:
        for annot in data_json['annotations']:
            if annot_by_imageid.get(annot['image_id'], 0) == 0:
                annot_by_imageid[annot['image_id']] = annot['caption'].strip('\n').lower()
    else:
        for annot in data_json['annotations']:
            if annot_by_imageid.get(annot['image_id'], 0):
                annot_by_imageid[annot['image_id']].append(annot['caption'].strip('\n').lower())
            else:
                annot_by_imageid[annot['image_id']] = []
                annot_by_imageid[annot['image_id']].append(annot['caption'].strip('\n').lower())
    return annot_by_imageid


def plot_images(df, img_path, num_images=5):
    df_tmp = df.sample(num_images).reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    
    for idx, row in df_tmp.iterrows():
        
        ax = plt.subplot(1, num_images, idx + 1)
        
        row_image = Image.open(os.path.join(img_path, row['filename']))
        
        caption = row['captions']
        caption = "\n".join(wrap(caption, 32))
        
        plt.title(caption)
        plt.imshow(row_image)
        plt.axis("off")
        
def get_transform(
        side_size = 256,
        mean = [0.45, 0.45, 0.45],
        std = [0.225, 0.225, 0.225],
        crop_size = 224,
        typeset = "train"
    ):
    if typeset == "train":
        return transforms.Compose([
                transforms.Resize(side_size),              
                transforms.RandomCrop(crop_size),                  
                transforms.RandomHorizontalFlip(),           
                transforms.ToTensor(),                       
                transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
                transforms.Resize(side_size),              
                transforms.RandomCrop(crop_size),                  
                transforms.RandomHorizontalFlip(),           
                transforms.ToTensor(),                       
                transforms.Normalize(mean, std)
        ])
        

def get_vocab(df, tokenizer):
    counter = Counter()
    for _, row in df.iterrows():
        words_de = row["captions"].lower() 
        counter.update(tokenizer(words_de))
    myvocab = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    myvocab.set_default_index(0)
    return myvocab

def load_images(captions_json, imgs_path):
    
    with open(os.path.join('data', captions_json['train']), 'r') as f:
        data_train = json.load(f)
    
    with open(os.path.join('data', captions_json['val']), 'r') as f:
        data_val = json.load(f)

    annot_by_imageid_train = get_annot_by_imageid(data_train, unique=True)
    annot_by_imageid_val = get_annot_by_imageid(data_val, unique=True)

    imgs_filename_train = os.listdir(os.path.join(imgs_path, 'train'))
    imgs_filename_train = [img_tmp for img_tmp in imgs_filename_train if img_tmp.endswith('jpg')]

    imgs_filename_val = os.listdir(os.path.join(imgs_path, 'val'))
    imgs_filename_val = [img_tmp for img_tmp in imgs_filename_val if img_tmp.endswith('jpg')]

    imgs_train = {
        'id':[],
        'filename':imgs_filename_train,
        'captions': []
    }
    for img_fl in imgs_filename_train:
        img_id = get_imageid(img_fl)
        imgs_train['id'].append(img_id)
        imgs_train['captions'].append(annot_by_imageid_train[int(img_id)])

    imgs_val = {
        'id':[],
        'filename':imgs_filename_val,
        'captions': []
    }
    for img_fl in imgs_filename_val:
        img_id = get_imageid(img_fl)
        imgs_val['id'].append(img_id)
        imgs_val['captions'].append(annot_by_imageid_val[int(img_id)])
    
    df_train = pd.DataFrame(imgs_train)
    df_val = pd.DataFrame(imgs_val)

    return df_train, df_val