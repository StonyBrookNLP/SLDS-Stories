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

from SLDS import SLDS
#from SLDS_bias import SLDS

from Sampler import GibbsSampler

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

    test_trans_matrix = torch.Tensor([[0.39, 0.41, 0.20],
                                  [0.31, 0.38, 0.31],
                                  [0.20, 0.41, 0.39]])



    #Load the data
    print("\nLoading Vocab")
    vocab = du.load_vocab(args.vocab)
    print("Vocab Loaded, Size {}".format(len(vocab.stoi.keys())))

    label_vocab = du.sentiment_label_vocab()

    print("Loading Dataset")
   # dataset = du.RocStoryDatasetSentiment(args.train_data, vocab, label_vocab) 
    dataset = du.RocStoryDataset(args.train_data, vocab, test=True) 
    print("Finished Loading Dataset {} examples".format(len(dataset)))


#    story_batches = du.RocStoryBatches(dataset, args.batch_size, sort_key=sort_func, train=True, sort_within_batch=True, device=-1)
    story_batches = du.RocStoryBatches(dataset, 1, train=False, sort=False, device=-1)
    data_len = len(dataset)

    print("Loading the Model")
    model = torch.load(args.load_model, map_location='cpu')
    model.set_use_cuda(use_cuda)
    model.eval()

    eps = [torch.randn(1, model.hidden_size) for _ in range(5)]

    sampler = GibbsSampler(model, vocab, use_bias=args.bias)

    for iteration, story in enumerate(story_batches): #this will continue on forever (shuffling every epoch) till epochs finished
        batch, seq_lens = story_batches.combine_story(story) #should return batch tensor [num_sents, batch, seq_len] and seq_lens [num_sents, batch]
       # state_targets= story_batches.combine_sentiment_labels(story, use_cuda=use_cuda) #state_targets is [numsents, batch, 1]
        state_targets= None

        if use_cuda:
            batch = batch.cuda()

        outputs = sampler.aggregate_gibbs_sample(batch, state_targets, [1,2,3], seq_lens, 1, 250, test_trans_matrix) 
#        outputs = sampler.aggregate_gibbs_sample(batch, state_targets, [0,2,4], seq_lens, 1, 250, test_trans_matrix) 
        print(outputs)
#        for i, sent in enumerate(outputs):
#            print("TRUE: {}".format(transform(batch[i].data.squeeze(), vocab.itos)))
#            print("{}".format(transform(outputs[i], vocab.itos)))
#        print("--------------------\n\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SLDS')
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--vocab', type=str, help='the vocabulary pickle file', default='./data/rocstory_vocab_f5.pkl')
    parser.add_argument('--seed', type=int, default=11, help='random seed') 
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('-bias', action='store_true', help='use bias')
    parser.add_argument('-src_seq_length', type=int, default=50, help="Maximum source sequence length")
    parser.add_argument('-max_decode_len', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('--initial_sents', type=int, default=0)
    parser.add_argument('--load_model', type=str)
    
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



