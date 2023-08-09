import torch
from torch.utils.data import DataLoader
import os
from model import ImageCaptionModel
from utils import *
from data import ImageCaptionDataset, generate_batch
import matplotlib.pyplot as plt

checkpoint_path = "checkpts"
data_path = 'data'
imgs_path = "data/img"
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 1
embed_size = 512         
hidden_size = 512  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


checkpoint = torch.load(os.path.join(checkpoint_path, 'final_model1.pth'))



captions_json = {f.replace('captions_','').replace('2014.json',''):f for f in os.listdir(data_path) if f.startswith('caption')}
df_train, df_val = load_images(captions_json, imgs_path)    
transform_train = get_transform(mean=mean, std=std)

dataset_train = ImageCaptionDataset(transform=transform_train, 
                            df=df_train, 
                            img_path= os.path.join(imgs_path, 'train')
                            )

train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle=True, collate_fn=generate_batch)
myvocab = train_loader.dataset.vocab
vocab_size = len(train_loader.dataset.vocab)

model = ImageCaptionModel(embed_size, hidden_size, vocab_size)
model.load_state_dict(checkpoint['model_state'])
model.eval()

loss = checkpoint['train_loss']

plt.figure(figsize=(10, 7))
plt.plot(loss, color='orange', linestyle='-', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over the epochs')
plt.savefig(os.path.join('checkpts', f'loss{2}.png'))




def get_text_from_ids(output):
    labels = []
    
    for idx in output:
        list_string.append(myvocab[idx])
    
    list_string = list_string[1:-1] 
    sentence = ' '.join(list_string)
    sentence = sentence.capitalize()
    return sentence

with torch.no_grad():
    image_orig, caption = next(iter(train_loader))

    image_pred = image_orig.unsqueeze(0).to(device)
    output = model(image_pred)
    print(output.shape)    
    #sentence = get_text_from_ids(output)
    
    
    plt.imshow(image_orig.permute(1,2,0))
    plt.title('Sample Image')
    plt.show()
    
