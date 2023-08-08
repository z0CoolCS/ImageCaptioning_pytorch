import wget
import zipfile
from tqdm.auto import tqdm
import os
from textwrap import wrap
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import json

def get_imageid(filename):
    id_image = filename.split('_')[-1].split('.')[0]
    return id_image.lstrip('0')

def get_annot_by_imageid(data_json, unique=False):
    annot_by_imageid = {}
    if unique:
        for annot in data_json['annotations']:
            if annot_by_imageid.get(annot['image_id'], 0) == 0:
                annot_by_imageid[annot['image_id']] = annot['caption'].strip('\n')
    else:
        for annot in data_json['annotations']:
            if annot_by_imageid.get(annot['image_id'], 0):
                annot_by_imageid[annot['image_id']].append(annot['caption'].strip('\n'))
            else:
                annot_by_imageid[annot['image_id']] = []
                annot_by_imageid[annot['image_id']].append(annot['caption'].strip('\n'))
    return annot_by_imageid

def download_captions():
    print('Downloading JSON files - captions')
    remote_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    
    local_file = 'data/annotations_trainval2014.zip'
    
    if not os.path.exists(local_file):
        wget.download(remote_url, local_file)

    with zipfile.ZipFile(local_file, 'r') as zip:
        zip.extractall('data')
    for file_json in os.listdir(os.path.join('data', 'annotations')):
        shutil.move(
            os.path.join('data', 'annotations', file_json), 
            os.path.join('data', file_json))

def download_images(data_json, annot_by_imageid, number_images=82783):
    print('Downloading JPG files - images')
    imgs = {
        'id':[],
        'filename':[],
        'captions': []
    }
    imgs_path = "data/img"

    for img in tqdm(data_json['images'][:number_images]):
        try:
            remote_url = img['coco_url']
            local_file = os.path.join(imgs_path, img['file_name'])
            if not os.path.exists(local_file):
                wget.download(remote_url, local_file)
                imgs['filename'].append(img['file_name'])
                imgs['id'].append(img['id'])
                imgs['captions'].append(annot_by_imageid[int(img['id'])])
        except:
            pass
        
    return imgs


#import numpy as np


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

if __name__ == '__main__':
    data_path = 'data'
    imgs_path = "data/img"
    download_captions()
    captions_json = {f.replace('captions_','').replace('2014.json',''):f for f in os.listdir(data_path) if f.startswith('caption')}
    with open(os.path.join('data', captions_json['train']), 'r') as f:
        data_train = json.load(f)
    annot_by_imageid = get_annot_by_imageid(data_train, unique=True)
    _ = download_images(data_train, annot_by_imageid)

