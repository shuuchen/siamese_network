from models import SiameseNetwork, TripletNetwork
from datasets import MadoriDataset, TriMadoriDataset
from loss import ContrastiveLoss, TripletLoss
from options import Config
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
import torch
import pandas as pd


# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
train_dataset = MadoriDataset() if Config.network == 'siamese' else TriMadoriDataset()
val_dataset = MadoriDataset(train=False) if Config.network == 'siamese' else TriMadoriDataset(train=False)

# data loaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=Config.batch_size)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=Config.batch_size)


# models
net = SiameseNetwork() if Config.network == 'siamese' else TripletNetwork()
net = net.to(device)
criterion = ContrastiveLoss() if Config.network == 'siamese' else TripletLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)


def train_siamese():
    train_loss_history, val_loss_history = [], []
    lowest_epoch_train_loss = lowest_epoch_val_loss = float('inf')

    for epoch in tqdm(range(Config.train_number_epochs)):
        # training
        net.train()
        epoch_train_loss = 0
        for batch_no, data in enumerate(train_dataloader):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device) , label.to(device)
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            batch_train_loss = criterion(output1,output2,label)
            epoch_train_loss += batch_train_loss.item()
            batch_train_loss.backward()
            optimizer.step()
        epoch_train_loss /= (batch_no + 1)
        if epoch_train_loss < lowest_epoch_train_loss:
            lowest_epoch_train_loss = epoch_train_loss
            #torch.save(net.state_dict(), f'{Config.checkpoint_dir}/best_train.pth')
        train_loss_history.append(epoch_train_loss)

        # validation
        net.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            for batch_no, data in enumerate(val_dataloader):
                img0, img1, label = data
                img0, img1, label = img0.to(device), img1.to(device) , label.to(device)
                output1,output2 = net(img0,img1)
                batch_val_loss = criterion(output1,output2,label)
                epoch_val_loss += batch_val_loss.item()
            epoch_val_loss /= (batch_no + 1)
            if epoch_val_loss < lowest_epoch_val_loss:
                lowest_epoch_val_loss = epoch_val_loss
                torch.save(net.state_dict(), f'{Config.checkpoint_dir}/best_val_siamese.pth')
            val_loss_history.append(epoch_val_loss)

        print(f'Epoch {epoch} training loss {epoch_train_loss}, validation loss {epoch_val_loss}')

    df = pd.DataFrame({'train_loss': train_loss_history, 'val_loss': val_loss_history})
    df.to_csv('./output/train_val_loss_siamese.csv')
    
    
def train_triplet():
    train_loss_history, val_loss_history = [], []
    lowest_epoch_train_loss = lowest_epoch_val_loss = float('inf')

    for epoch in tqdm(range(Config.train_number_epochs)):
        # training
        net.train()
        epoch_train_loss = 0
        for batch_no, data in enumerate(train_dataloader):
            anchor, neg, pos = data
            anchor, neg, pos = anchor.to(device), neg.to(device) , pos.to(device)
            optimizer.zero_grad()
            anchor_vec, neg_vec, pos_vec = net(anchor, neg, pos)
            batch_train_loss = criterion(anchor_vec, neg_vec, pos_vec)
            epoch_train_loss += batch_train_loss.item()
            batch_train_loss.backward()
            optimizer.step()
        epoch_train_loss /= (batch_no + 1)
        if epoch_train_loss < lowest_epoch_train_loss:
            lowest_epoch_train_loss = epoch_train_loss
        train_loss_history.append(epoch_train_loss)

        # validation
        net.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            for batch_no, data in enumerate(val_dataloader):
                anchor, neg, pos = data
                anchor, neg, pos = anchor.to(device), neg.to(device) , pos.to(device)
                anchor_vec, neg_vec, pos_vec = net(anchor, neg, pos)
                batch_val_loss = criterion(anchor_vec, neg_vec, pos_vec)
                epoch_val_loss += batch_val_loss.item()
            epoch_val_loss /= (batch_no + 1)
            if epoch_val_loss < lowest_epoch_val_loss:
                lowest_epoch_val_loss = epoch_val_loss
                torch.save(net.state_dict(), f'{Config.checkpoint_dir}/best_val_triplet.pth')
            val_loss_history.append(epoch_val_loss)

        print(f'Epoch {epoch} training loss {epoch_train_loss}, validation loss {epoch_val_loss}')

    df = pd.DataFrame({'train_loss': train_loss_history, 'val_loss': val_loss_history})
    df.to_csv('./output/train_val_loss_triplet.csv')
    
if __name__ == '__main__':
    if Config.network == 'siamese':
        train_siamese()
    else:
        train_triplet()