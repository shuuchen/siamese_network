import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.distance import PairwiseDistance


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - 
                                        euclidean_distance, min=0.0), 2))
        return loss_contrastive

    
class TripletLoss(nn.Module):
    
    def __init__(self, margin=5.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist  = PairwiseDistance(2)
        
    def forward(self, anchor, negative, positive):
        pos_dist   = self.pdist.forward(anchor, positive)
        neg_dist   = self.pdist.forward(anchor, negative)
        
        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min = 0.0)
        loss       = torch.mean(hinge_dist)
        return loss