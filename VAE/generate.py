###########################################
# Model for generating samples from model
#
###########################################
import torch 
import torch.nn as nn
from torchtext.data import Iterator as BatchIter
import argparse
import numpy as np
import math
import time
from torch.autograd import Variable
import torch.nn.functional as F
import data_utils as du
import random

from data_utils import EOS_TOK, SOS_TOK, PAD_TOK, TUP_TOK, MAX_EVAL_SEQ_LEN, MIN_EVAL_SEQ_LEN
#from decode_utils import transform, get_tups, get_pred_events

import sys

from model import *
import data_utils as du


USE_CUDA=False
hidden_size = 512
embed_size = 300 
learning_rate = 0.0001
n_epochs = 100000
grad_clip = 1.0

kld_start_inc = 10000
kld_weight = 0.05
kld_max = 0.1
kld_inc = 0.000002
temperature = 0.9
temperature_min = 0.5
temperature_dec = 0.000002



def generate(args):
    """
    Use the trained model for decoding
    Args
        args (argparse.ArgumentParser)
    """
    if args.cuda and torch.cuda.is_available():
        device = 0
        use_cuda = True
    elif args.cuda and not torch.cuda.is_available():
        print("You do not have CUDA, turning cuda off")
        device = -1 
        use_cuda = False
    else:
        device = -1
        use_cuda=False

    #Load the vocab
    vocab = du.load_vocab(args.vocab)
    eos_id = vocab.stoi[EOS_TOK]
    pad_id = vocab.stoi[PAD_TOK]

    if args.ranking: # default is HARD one
        dataset = du.NarrativeClozeDataset(args.valid_data, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN, LM=False)
        # Batch size during decoding is set to 1
        batches = BatchIter(dataset, 1, sort_key=lambda x:len(x.actual), train=False, device=device)
    else:
        dataset = du.SentenceDataset(args.valid_data, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN, add_eos=False) #put in filter pred later
        # Batch size during decoding is set to 1
        batches = BatchIter(dataset, 1, sort_key=lambda x:len(x.text), train=False, device=device)

    data_len = len(dataset)

    #Create the model
    with open(args.load, 'rb') as fi:
        model = torch.load(fi, map_location=lambda storage, loc : storage)

    model.decoder.eval()
    model.encoder.eval()

    #For reconstruction
    if args.perplexity:
        loss = calc_perplexity(args, model, batches, vocab, data_len)
        print("Loss = {}".format(loss))
    elif args.ranking:
        do_ranking(args, model, batches, vocab, data_len, use_cuda)


def do_ranking(args, model, batches, vocab, data_len, use_cuda):
    print("RANKING")
    ranked_acc = 0.0

    tup_idx = vocab.stoi[TUP_TOK]
    
    
    for iteration, bl in enumerate(batches):

        if (iteration+1)%25 == 0:
            print("iteration {}".format(iteration+1))

        all_texts = [bl.actual, bl.actual_tgt, bl.dist1, bl.dist1_tgt, bl.dist2, bl.dist2_tgt, bl.dist3, bl.dist3_tgt, bl.dist4, bl.dist4_tgt, bl.dist5, bl.dist5_tgt] # each is a tup

        assert len(all_texts) == 12, "12 = 6 * 2."

        all_texts_vars = []
        if use_cuda:    
            for tup in all_texts:
                all_texts_vars.append((Variable(tup[0].cuda(), volatile=True), tup[1]))
        else:
            for tup in all_texts:
                all_texts_vars.append((Variable(tup[0], volatile=True), tup[1]))

        # will itetrate 2 at a time using iterator and next
        vars_iter = iter(all_texts_vars)
        # run the model for all 6 sentences
        pps = []

        first_tup = -1 
        for i in range(bl.actual[0].shape[1]):
            if bl.actual[0][0, i] == tup_idx:
                first_tup = i
                break
        if first_tup == -1: 
            print("WARNING: First TUP is -1")
        src_tup = Variable(bl.actual[0][:, :first_tup+1].view(1, -1), volatile=True)
        src_lens = torch.LongTensor([src_tup.shape[1]])

        dhidden, latent_values = model(src_tup, src_lens, encode_only=True)
        
        # Latent and hidden have been initialized with the first tuple
        for tup in vars_iter:
            ## INIT FEED AND DECODE before every sentence. 
            model.decoder.init_feed_(Variable(torch.zeros(1, model.decoder.attn_dim)))

            next_tup = next(vars_iter)
            _, _, _, dec_outputs  = model.train(tup[0], 1, dhidden, latent_values, []) 
            logits = model.logits_out(dec_outputs)
            logits = logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]
            nll = masked_cross_entropy(logits, next_tup[0], Variable(next_tup[1]))
            #nll = calc_perplexity(args, model, tup[0], vocab, next_tup[0], next_tup[1], hidden) 
            pp = torch.exp(nll)
            #print("NEG-LOSS {} PPL {}".format(nll.data[0], pp.data[0]))
            pps.append(pp.data.numpy()[0])

        # low perplexity == top ranked sentence- correct answer is the first one of course
        assert len(pps) == 6, "6 targets."
        #print("\n")
        all_texts_str = [transform(text[0].data.numpy()[0], vocab.itos) for text in all_texts_vars]
        #print("ALL: {}".format(all_texts_str))
        min_index = np.argmin(pps)
        if min_index == 0:
            ranked_acc += 1
            #print("TARGET: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
            #print("CORRECT: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
        #else:
            # print the ones that are wrong
            #print("TARGET: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
            #print("WRONG: {}".format(transform(all_texts_vars[min_index+2][0].data.numpy()[0], vocab.itos)))

        if (iteration+1) == args.max_decode:
            print("Max decode reached. Exiting.")
            break

    ranked_acc /= (iteration+1) * 1/100 # multiplying to get percent 
    print("Average acc(%): {}".format(ranked_acc))


def calc_perplexity(args, model, batches, vocab, data_len):
    total_loss = 0.0
    iters = 0
    for iteration, bl in enumerate(batches):
        print(iteration)
        
        batch, batch_lens = bl.text
        target, target_lens = bl.target 

        target = Variable(target.squeeze(0), volatile=True)

        batch = Variable(batch.squeeze(0), volatile=True)
        m, l, z, decoded =model(batch)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(decoded, target)

        total_loss = total_loss + loss.data[0]

        iters += 1

    print(iters)
    print(data_len)

    return total_loss / data_len





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DAVAE')
    parser.add_argument('--impute_with', type=int, default=0)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--min_vocab_freq', type=int, default=1)
    parser.add_argument('--vocab', type=str) 
    parser.add_argument('--emb_size', type=int, default=200, help='size of word embeddings')
    parser.add_argument('--hid_size', type=int, default=200,help='size of hidden')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--log_after', type=int, default=200)
    parser.add_argument('--save_after', type=int, default=500)
    parser.add_argument('--validate_after', type=int, default=2500)
    parser.add_argument('--optimizer', type=str, default='adagrad', help='adam, adagrad, sgd')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--load', type=str,  default='model.pt',help='path to load the final model')
    parser.add_argument('--bidir', type=bool, default=True, help='Use bidirectional encoder')
    parser.add_argument('--latent', type=str, help='A str in form of python list')
    parser.add_argument('--beam_size',  type=int, default=-1, help='Beam size')
    parser.add_argument('-perplexity',  action='store_true')
    parser.add_argument('-schema',  action='store_true')
    parser.add_argument('-max_len_decode', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('--n_best', type=int, default=1, help="""outputs the n_best decoded sentences""")
    parser.add_argument('--ranking',  action='store_true', help="""N cloze ranking""")
    parser.add_argument('--max_decode', type=int, default=2000, help="""max sentences to be evaluated/decoded.""")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    generate(args)



