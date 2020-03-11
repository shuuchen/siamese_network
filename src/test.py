from models import SiameseNetwork, TripletNetwork
from datasets import MadoriDataset, TriMadoriDataset
from loss import ContrastiveLoss, TripletLoss
from options import Config
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch
import pandas as pd


epoch_test = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = TripletNetwork().to(device)
net.load_state_dict(torch.load(f'{Config.checkpoint_dir}/best_val_triplet.pth'))
test_dataloader = DataLoader(TriMadoriDataset(test=True), 
                             shuffle=False, 
                             batch_size=1)
print(len(test_dataloader))

net.eval()
test_results_0 = []
test_results_1 = []
with torch.no_grad():
    for batch_no, data in enumerate(test_dataloader):
        
        anchor, neg, pos = data
        anchor, neg, pos = anchor.to(device), neg.to(device) , pos.to(device)
        anchor_vec, neg_vec, pos_vec = net(anchor, neg, pos)
        
        dis_anchor_neg = F.pairwise_distance(anchor_vec, neg_vec)
        dis_anchor_pos = F.pairwise_distance(anchor_vec, pos_vec)
        
        print(f'anchor_neg: {dis_anchor_neg.item()}, anchor_pos: {dis_anchor_pos.item()}')
        
        test_results_1 += [dis_anchor_neg.item()]
        test_results_0 += [dis_anchor_pos.item()]

df = pd.DataFrame({'test_results_0': test_results_0, 'test_results_1': test_results_1})

fig = df.plot(grid=True, title='triplet test results').get_figure()
fig.savefig('./output/triplet/test_result.png')
