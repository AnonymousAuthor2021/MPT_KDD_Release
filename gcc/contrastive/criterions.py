import torch
from torch import nn
import math
import torch.nn.functional as F
class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        # bsz = x.shape[0]
        x = x.squeeze()
        # print(x.shape, label.shape)
        # label = torch.zeros([bsz]).cuda().long()
        label = label.cuda().long()
        loss = self.criterion(x, label)
       
        return loss


class NCESoftmaxLossNS(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLossNS, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()
        # L1 loss, L2 loss
        self.criterion = nn.L1Loss()
        #self.criterion = nn.MSELoss()
    def forward(self, x, label):
        #print(x, label)
        #exit(0)
        # bsz = x.shape[0]
        x = x.squeeze()
        # positives on the diagonal
        # label = torch.arange(bsz).cuda().long()
        # print(self.criterion)
        label = label.cuda().float()
        loss = self.criterion(x, label)

        #print(x.size())
        #print(x, label, loss)
        return loss

class NCESoftmaxLossNS2(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLossNS2, self).__init__()
        self.loss_fn = nn.MSELoss()

    def myloss(self, input, target):
        first1 = 0
        second1 = 0
        for i in range(target.size(0)):
            first1 += -input[i][target[i]]
            tempSum = 0
            for j in range(input.size(1)):
                tempSum+=math.exp(input[i][j])
            second1+=math.log(tempSum)
        return ((first1+second1)/input.size(0))
    
    def myloss2(self, input, target):
        input = F.softmax(input)
        return self.loss_fn(input, torch.eye(target.shape[0]).cuda())
    
    def softmax(self, x, sim):
        exp_x = torch.exp(x)
        #attention = torch.ones(x.shape[0], x.shape[1]).cuda()
        attention = torch.from_numpy(sim).cuda()
        exp_x2 = attention.mul(exp_x)
        #print(sim[0][0])
        # print(exp_x)
        # exit(0)
        sum_x = torch.sum(exp_x2, dim=1, keepdim=True)
        return exp_x/sum_x

    def log_softmax(self, x, sim):
        return torch.log(self.softmax(x, sim))

    def myloss3(self, outputs, targets, sim):
        num_examples = targets.shape[0]
        batch_size = outputs.shape[0]

        outputs = self.log_softmax(outputs, sim)
        targets = targets.view(-1, 1)
        one_hot = torch.zeros(outputs.shape[0], outputs.shape[1]).cuda().scatter_(1, targets, 1)
        outputs = one_hot.mul(outputs)
        outputs = torch.sum(outputs, dim=1)
        return -torch.sum(outputs)/num_examples
    
    def forward(self, x, sim):
        bsz = x.shape[0]
        x = x.squeeze()
        # positives on the diagonal
        label = torch.arange(bsz).cuda().long()
        loss = self.myloss3(x, label, sim)
        return loss
