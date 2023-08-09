import json
import os
import shutil
import zipfile
import wget
from tqdm.auto import tqdm

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

def download_images(data_json, folder,number_images=82783):

    imgs_path = os.path.join("data/img", folder)

    for img in tqdm(data_json['images'][:number_images]):
        try:
            remote_url = img['coco_url']
            local_file = os.path.join(imgs_path, img['file_name'])
            if not os.path.exists(local_file):
                wget.download(remote_url, local_file)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    data_path = 'data'
    imgs_path = "data/img"
    download_captions()
    captions_json = {f.replace('captions_','').replace('2014.json',''):f for f in os.listdir(data_path) if f.startswith('caption')}

    with open(os.path.join('data', captions_json['train']), 'r') as f:
        data_train = json.load(f)

    print('Downloading Training JPG files - images')
    download_images(data_train, 'train')

    with open(os.path.join('data', captions_json['val']), 'r') as f:
        data_val = json.load(f)

    print('Downloading Validation JPG files - images')
    download_images(data_val, 'val', len(data_val['images']))
