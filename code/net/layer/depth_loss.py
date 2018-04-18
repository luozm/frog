import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def depth_loss(logits, labels, instances):
    #print(logits.size())
    #print(labels.size())
    #print(instances.size())
    #return 1
    batch_size, num_classes = logits.size(0), logits.size(1)

    logits_flat = logits.view(batch_size, num_classes, -1)
    dim = logits_flat.size(2)

    # one hot encode
#    select = Variable(torch.zeros((batch_size, num_classes))).cuda()
#    select.scatter_(1, labels.view(-1, 1), 1)
#    select[:, 0] = 0
#    select = select.view(batch_size, num_classes, 1).expand((batch_size, num_classes, dim)).contiguous().byte()

#    logits_flat = logits_flat[select].view(-1)
    logits_flat = logits_flat.view(-1)
    labels_flat = instances.view(-1)
#    print(logits_flat.size())
#    print(labels_flat.size())

    loss = F.mse_loss(logits_flat, labels_flat)
#    loss = Variable(torch.cuda.FloatTensor(1).zero_()).sum()
    #loss = ((logits_flat-labels_flat)**2).mean()

    return loss


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
