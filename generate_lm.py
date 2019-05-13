########################################
#   module for training - incomplete
########################################
import torch 
import torch.nn as nn
from torchtext.data import Iterator as BatchIter
import argparse
import numpy as np
import random
import math
from torch.autograd import Variable
from EncDec import Encoder, Decoder
import torch.nn.functional as F
import data_utils as du
from LM import LM

from masked_cross_entropy import masked_cross_entropy
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK, transform
import time
from torchtext.vocab import GloVe
import pickle
import gc
import glob
import sys
import os



def check_save_model_path(save_model):
    save_model_path = os.path.abspath(save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

        
def generate(args):
    """
    Train the model in the ol' fashioned way, just like grandma used to
    Args
        args (argparse.ArgumentParser)
    """
    if args.cuda and torch.cuda.is_available():
        print("Using cuda")
        use_cuda = True
    elif args.cuda and not torch.cuda.is_available():
        print("You do not have CUDA, turning cuda off")
        use_cuda = False
    else:
        use_cuda=False

    
    #Load the data
    print("\nLoading Vocab")
    vocab = du.load_vocab(args.vocab)
    print("Vocab Loaded, Size {}".format(len(vocab.stoi.keys())))

    print("Loading Dataset")
    dataset = du.RocStoryDataset(args.train_data, vocab, test=True) 
#    dataset = du.RocStoryDataset(args.train_data, vocab, test=False) 
    print("Finished Loading Dataset {} examples".format(len(dataset)))

    sort_func = lambda x: max([len(x.sent_1),len(x.sent_2),len(x.sent_3),len(x.sent_4),len(x.sent_5)])

#    story_batches = du.RocStoryBatches(dataset, args.batch_size, sort_key=sort_func, train=True, sort_within_batch=True, device=-1)
    story_batches = du.RocStoryBatches(dataset, 1, train=False, sort=False, device=-1)
    data_len = len(dataset)

    #0-antici, 1-anger, 2-disgust, 3-sad, 4-suprise, 5-fear, 6-trust, 7-joy
    test_trans_matrix = torch.Tensor([[0.10, 0.10, 0.05, 0.10, 0.25, 0.10,0.10, 0.20],
                                      [0.10, 0.10, 0.20, 0.25, 0.10, 0.10,0.10, 0.05],
                                      [0.10, 0.25, 0.20, 0.10, 0.10, 0.10,0.10, 0.05],
                                      [0.10, 0.20, 0.25, 0.10, 0.10, 0.10,0.10, 0.05],
                                      [0.25, 0.10, 0.10, 0.10, 0.10, 0.10,0.05, 0.20],
                                      [0.10, 0.10, 0.10, 0.20, 0.25, 0.10,0.05, 0.10],
                                      [0.20, 0.10, 0.10, 0.10, 0.05, 0.10,0.10, 0.25],
                                      [0.20, 0.05, 0.10, 0.10, 0.10, 0.10,0.25, 0.10]])

    print("Loading the Model")
    model = torch.load(args.load_model, map_location='cpu')
    model.set_use_cuda(use_cuda)
    model.eval()

    # calculate NLL
    if args.nll:
        train_loss = 0.0
        for iteration, story in enumerate(story_batches):    
            batch, seq_lens = story_batches.combine_story(story)
            targets, target_lens = story_batches.convert_to_target(batch, seq_lens)

            if use_cuda:
                batch = batch.cuda()
                targets = targets.cuda()
                targen_lens= target_lens.cuda()

            text_logits = model(batch, seq_lens, gumbel_temp=gumbel_temp)
            loss, _ = compute_loss_unsupervised(text_logits, targets, target_lens, iteration, use_cuda=use_cuda)
            total_loss += loss.item()
        nll = total_loss / data_len # batch_size is one

        print("NLL {:.4f}".format(nll))

        break # break out

            
    for iteration, story in enumerate(story_batches): 
        batch, seq_lens = story_batches.combine_story(story) #should return batch tensor [num_sents, batch, seq_len] and seq_lens [num_sents, batch]

        if use_cuda:
            batch = batch.cuda()

        if args.interpolate:
            outputs = model.interpolate(batch, seq_lens, args.initial_sents, args.num_samples, vocab)
        else:
            outputs = model.reconstruct(batch, seq_lens, args.initial_sents)

        
        for i, sent in enumerate(outputs):
            #print("TRUE: {}".format(transform(batch[i].data.squeeze(), vocab.itos)))
            print("{}".format(transform(outputs[i], vocab.itos)))
        print("==========================\n\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SLDS')
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--vocab', type=str, help='the vocabulary pickle file', default='./data/rocstory_vocab_f5.pkl')
    parser.add_argument('--seed', type=int, default=11, help='random seed') 
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('-src_seq_length', type=int, default=50, help="Maximum source sequence length")
    parser.add_argument('-max_decode_len', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('--initial_sents', type=int, default=1)
    parser.add_argument('-interpolate', action='store_true')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--nll', action='store_true', help='Calculate NLL value')    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # diff between train and classic: in classic pass .txt etension for files.
    generate(args)



