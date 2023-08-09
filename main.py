from utils import *
from data import ImageCaptionDataset, generate_batch
from model import ImageCaptionModel
from checkpoint import SaveModel
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


def traininig_stage(model, train_loader, optimizer, criterion, vocab_size):
    model.train()
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

    return np.mean(loss_epoch)

def validation_stage(model, val_loader, criterion, vocab_size):
    model.eval()
    loss_epoch = []
    with torch.no_grad():
        for images, captions in tqdm(val_loader):
            
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions)

            loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.contiguous().view(-1))
            
            loss_epoch.append(loss.item())

    return np.mean(loss_epoch)
    


def train_model(model, train_loader, val_loader, optimizer, criterion, vocab_size, epochs=20, save_model=SaveModel('checkpts'), version_model=1):
    loss_total_train = []
    loss_total_val = []
    for epoch in tqdm(range(1, epochs+1)):
        
        loss_epoch = traininig_stage(model, train_loader, optimizer, criterion, vocab_size)   
        loss_total_train.append(loss_epoch)

        loss_epoch = validation_stage(model, val_loader, criterion, vocab_size)   
        loss_total_val.append(loss_epoch)
        
        stats = '\nEpoch [%d], Loss Train: %.4f, Loss Val: %.4f \n' % (epoch, loss_total_train[-1], loss_total_val[-1])
        print(stats)

        save_model(
            current_valid_loss= loss_total_val[-1], 
            model=model, 
            optimizer=optimizer, 
            criterion=criterion, 
            epoch=epoch
            )
        
    save_model.save_model_final(
        epochs=epochs, 
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        train_loss=loss_total_train,
        valid_loss= loss_total_val, 
        index=version_model)    
    
    save_model.save_plots_loss(loss_total_train, loss_total_val, version_model)
            

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
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    save_model = SaveModel('checkpts')

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        vocab_size=vocab_size,
        epochs=20,
        save_model=save_model
        )

    

if __name__ == '__main__':
    main()

    

   

