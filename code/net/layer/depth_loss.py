import os
import torch
import torch.nn as nn
from torch.autograd import Variable


def depth_loss(logits, labels, instances):

    batch_size, num_classes = logits.size(0), logits.size(1)

    logits_flat = logits.view(batch_size, num_classes, -1)
    dim = logits_flat.size(2)

    # one hot encode
    select = Variable(torch.zeros((batch_size, num_classes))).cuda()
    select.scatter_(1, labels.view(-1, 1), 1)
    select[:, 0] = 0
    select = select.view(batch_size, num_classes, 1).expand((batch_size, num_classes, dim)).contiguous().byte()

    logits_flat = logits_flat[select].view(-1)
    labels_flat = instances.view(-1)

    loss = nn.MSELoss(logits_flat, labels_flat)
    return loss


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
