import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def weighted_cross_entropy_with_logits(logits, labels, weights):

    log_probs = F.log_softmax(logits)
    labels    = labels.view(-1, 1)
    loss = -torch.gather(log_probs, dim=1, index=labels)
    loss = weights*loss.view(-1)

    loss = loss.sum()/(weights.sum()+1e-12)
    return loss

def depth_loss(logits, labels, instances):

#    batch_size, num_classes = logits.size(0), logits.size(1)

#    logits_flat = logits.view(batch_size, num_classes, -1)
#    dim = logits_flat.size(2)

    # one hot encode
#    select = Variable(torch.zeros((batch_size, num_classes))).cuda()
#    select.scatter_(1, labels.view(-1, 1), 1)
#    select[:, 0] = 0
#    select = select.view(batch_size, num_classes, 1).expand((batch_size, num_classes, dim)).contiguous().byte()

#    logits_flat = logits_flat[select].view(-1)
    #logits_flat = logits_flat.view(-1).float()

    logits = logits.permute(0, 2, 3, 1).contiguous()
    logits_flat = logits.view(-1, 16)
    labels_flat = instances.view(-1).long()
    weights = [3.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    weights = torch.FloatTensor(weights).cuda()

    #loss = F.mse_loss(logits_flat, labels_flat)
    loss = F.cross_entropy(logits_flat, labels_flat, weight=weights)

    return loss


def dir_loss(logits, labels, instances):
    logits = logits.permute(0, 2, 3, 1).contiguous()
    logits_flat = logits.view(-1, 2)
#    logits_flat = logits_flat.view(-1).float()
    labels_flat = instances.view(-1, 2).float()

    errorAngles = torch.acos(torch.sum(logits_flat*labels_flat, 1))
    loss = torch.sum(errorAngles*errorAngles)/errorAngles.shape[0]
    return loss

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
