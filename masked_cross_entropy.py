#Masked Cross Entropy Functions 
#All credit here goes to https://github.com/jihunchoi
import torch 
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as functional 


def compute_loss_unsupervised_LM(text_logits, text_targets, target_lens, iteration, use_cuda=True, do_print=True): 
    """
    Compute the loss term 
    Args: 
    text_logits [num_sents, batch, seq_len, vocab size]
    text_targets [num_sents, batch, seq_len] : the id of the true class to predict
    target_lens [num_sents, batch] : the length of the targets
    Z_kl [batch]
    state_kl [batch]
    """

    num_sents = text_logits.shape[0]
    batch_size = text_logits.shape[1]

    
    #compute CE loss for data
    if use_cuda:
        ce_loss = Variable(torch.Tensor([0]).cuda())
    else:
        ce_loss = Variable(torch.Tensor([0]))

    for i in range(num_sents):
        ce_loss += masked_cross_entropy(text_logits[i], text_targets[i], target_lens[i], use_cuda=use_cuda)

    total_loss = ce_loss 
    
    if do_print:
        print_iter_stats(iteration, total_loss, ce_loss, 0, 0)
    
    return total_loss, ce_loss # tensor 
 

def compute_loss_unsupervised(text_logits, text_targets, target_lens, Z_kl, state_kl, iteration, kld_weight, use_cuda=True, do_print=True, last_loss=False): 
    """
    Compute the loss term 
    Args: 
    text_logits [num_sents, batch, seq_len, vocab size]
    text_targets [num_sents, batch, seq_len] : the id of the true class to predict
    target_lens [num_sents, batch] : the length of the targets
    Z_kl [batch]
    state_kl [batch]
    """

    num_sents = text_logits.shape[0]
    batch_size = text_logits.shape[1]

    Z_kl_mean = Z_kl.mean()
    state_kl_mean= state_kl.mean()
    
    #compute CE loss for data
    if use_cuda:
        ce_loss = Variable(torch.Tensor([0]).cuda())
    else:
        ce_loss = Variable(torch.Tensor([0]))

    # used this in ncloze
    if last_loss:
        last_ce_loss = masked_cross_entropy(text_logits[4], text_targets[4], target_lens[4], use_cuda=use_cuda)
        return last_ce_loss

    for i in range(num_sents):
        ce_loss += masked_cross_entropy(text_logits[i], text_targets[i], target_lens[i], use_cuda=use_cuda)

    total_loss = ce_loss + kld_weight*Z_kl_mean + state_kl_mean
    if do_print:
        print_iter_stats(iteration, total_loss, ce_loss, Z_kl_mean, state_kl_mean)
 
    return total_loss, ce_loss # tensor 


def compute_loss_supervised(text_logits, text_targets, target_lens, Z_kl, state_logits, state_targets, iteration, kld_weight, use_cuda=True, do_print=True): 
    """
    Compute the loss term 
    Args: 
    text_logits [num_sents, batch, seq_len, vocab size]
    text_targets [num_sents, batch, seq_len] : the id of the true class to predict
    target_lens [num_sents, batch] : the length of the targets
    state_logits [num_sents, batch, num_classes]
    state_targets [num_sents, batch]
    Z_kl [batch]
    """

    num_sents = text_logits.shape[0]
    batch_size = text_logits.shape[1]

    Z_kl_mean = Z_kl.mean()

    
    #compute CE loss for data
    if use_cuda:
        ce_loss = Variable(torch.Tensor([0]).cuda())
        state_loss = Variable(torch.Tensor([0]).cuda())
        state_loss_fn = torch.nn.CrossEntropyLoss().cuda()
    else:
        ce_loss = Variable(torch.Tensor([0]))
        state_loss = Variable(torch.Tensor([0]))
        state_loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(num_sents):
        ce_loss += masked_cross_entropy(text_logits[i], text_targets[i], target_lens[i], use_cuda=use_cuda)
        state_loss += state_loss_fn(state_logits[i], state_targets[i])

#    if Z_kl_mean <= 20.0: #Model_try2 useses free bits
#    if Z_kl_mean <= 1000.0 and iteration <= 10000:
#        print("Free Bits")
#        kld_weight = 0.0

    total_loss = ce_loss + kld_weight*Z_kl_mean + state_loss
    if do_print:
        print_iter_stats(iteration, total_loss, ce_loss, Z_kl_mean, state_loss)
    
    return total_loss, ce_loss, state_loss # tensor 
   


def print_iter_stats(iteration, total_loss, ce_loss, Z_kl, state_kl):

#    if iteration % args.log_every == 0 and iteration != 0:
    if True:
        print("Iteration: ", iteration) 
        print("Total: ", total_loss.cpu().data[0])
        print("CE: ", ce_loss.cpu().data[0])
        print("Z KL Div: ", Z_kl.cpu().data[0])
        print("State KL Div: ", state_kl.cpu().data[0])



def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long() #changed to arange
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def masked_cross_entropy(logits, target, length, use_cuda=True):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1) #added dim=1
    #print("log probs flat {}".format(log_probs_flat.size()))
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    #print("target flat {}".format(target_flat.size()))
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    #print("losses {}".format(losses.size))
    # mask: (batch, max_len)
    if use_cuda:
        mask = _sequence_mask(sequence_length=length, max_len=target.size(1)).cuda()
    else:
        mask = _sequence_mask(sequence_length=length, max_len=target.size(1)) 
    #print("mask {}".format(mask))
    losses = losses * mask.float()
    #print("losses {}".format(losses))
    loss = losses.sum() 

#    if shard:
#        return loss, length.float().sum() # changed it to return loss and length

    if use_cuda:
        return loss / length.cuda().float().sum() # average loss 
    else:
        return loss / length.float().sum() # average loss 
