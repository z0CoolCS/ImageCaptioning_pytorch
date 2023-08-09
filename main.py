import json
from utils import *

data_path = 'data'
imgs_path = "data/img"

def main():
    captions_json = {f.replace('captions_','').replace('2014.json',''):f for f in os.listdir(data_path) if f.startswith('caption')}
    df_train, df_val = load_images(captions_json, imgs_path)
    print(df_train.shape)
    print(df_val.shape)

if __name__ == '__main__':
    main()

    

   

