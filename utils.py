##############################################
# Random Util Stuff
##############################################
import torch 
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F

def normal_sample(mu, logvar, use_cuda=False):
    """
    Reparmaterization trick for normal distribution
    """ 
    eps = Variable(torch.randn(mu.size()))
    if use_cuda:
        eps = eps.cuda()
    std = torch.exp(logvar / 2.0)
    return mu + eps * std


def gumbel_sample(shape, use_cuda=False, eps=1e-20):
    """
    Sample from gumbel distribution
    """
    U = torch.rand(shape).cuda() if use_cuda else torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temp, use_cuda=False):
    """
    Sample from a gumbel softmax distribution 
    Args:
        logits (Tensor [batch, num classes])
    """
    y = logits + gumbel_sample(logits.size(), use_cuda=use_cuda)
    return F.softmax(y / temp, dim=1)

def get_context_vector(encoded_sents, target, future=False, use_cuda=False):
    """
    Get create the context vector for the sentence given at index target for 
    state classification. Do this by max pooling the sentences before the target sentence
    Args:
        encoded_sents (Tensor, [num_sents, batch, encoder dim])
        target (int) : Index of the target sentence (starts at 0)
        future (bool) : If true, use the vectors from the future instead of the past 
    Ret:
        context vector (Tensor [batch, encoder dim])
    """

    if target == 0 and not future:
        #return encoded_target[0, :, :]
        return torch.zeros(encoded_sents[0,:,:].shape).cuda() if use_cuda else torch.zeros(encoded_sents[0,:,:].shape)
    elif target == encoded_sents.shape[0]-1 and future:
        #return encoded_target[encoded_sents.shape[0]-1, :, :]
        return torch.zeros(encoded_sents[0,:,:].shape).cuda() if use_cuda else torch.zeros(encoded_sents[0,:,:].shape)


    if not future:
        sents = encoded_sents[:target, :, :] #[sents, batch, encoder dim]
    else:
        sents = encoded_sents[target+1:, :, :] #[sents, batch, encoder dim]

#    maxpool, _ = torch.max(sents, dim=0) #[batch, encoder dim]
    maxpool = torch.mean(sents, dim=0) #[batch, encoder dim]

    return maxpool

