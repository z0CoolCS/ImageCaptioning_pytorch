from utils import *
from data import ImageCaptionDataset, generate_batch
from model import ImageCaptionModel
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm.auto import tqdm


data_path = 'data'
imgs_path = "data/img"
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 4
embed_size = 512         
hidden_size = 512  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, optimizer, criterion, vocab_size, epochs=20):
    loss_total = []
    for epoch in tqdm(range(1, epochs+1)):
        loss_epoch = []
        for images, captions in tqdm(train_loader):
            
            images = images.to(device)
            captions = captions.to(device)
                       
            optimizer.zero_grad()
            
            outputs = model(images, captions)

            loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.contiguous().view(-1))
            
            loss.backward()
            
            optimizer.step()
            
            loss_epoch.append(loss.item())
            
        loss_total.append(np.mean(loss_epoch))
        stats = 'Epoch [%d], Loss: %.4f,' % (epoch, loss_total[-1])
        print(stats)
            

def main():

    captions_json = {f.replace('captions_','').replace('2014.json',''):f for f in os.listdir(data_path) if f.startswith('caption')}
    df_train, df_val = load_images(captions_json, imgs_path)    
    transform_train = get_transform(mean=mean, std=std)
    transform_val = get_transform(mean=mean, std=std)

    dataset_train = ImageCaptionDataset(transform=transform_train, 
                              df=df_train, 
                              img_path= os.path.join(imgs_path, 'train')
                              )
    dataset_val = ImageCaptionDataset(transform=transform_val, 
                              df=df_val, 
                              img_path= os.path.join(imgs_path, 'val')
                              )
    
    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle=True, collate_fn=generate_batch)
    val_loader = DataLoader(dataset_val, batch_size = batch_size, collate_fn=generate_batch)
    vocab_size = len(train_loader.dataset.vocab)

    criterion = torch.nn.CrossEntropyLoss() #.cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    model = ImageCaptionModel(embed_size, hidden_size, vocab_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    train(model, train_loader, val_loader, optimizer, criterion, vocab_size, 100)

    

if __name__ == '__main__':
    main()

    

   

