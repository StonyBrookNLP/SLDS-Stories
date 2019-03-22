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
from SLDS import SLDS
from masked_cross_entropy import masked_cross_entropy
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK
import time
from torchtext.vocab import GloVe
import pickle
import gc
import glob
import sys
import os

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)



def compute_loss(text_logits, text_targets, target_lens, Z_kl, state_kl, iteration, use_cuda=True): 
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

    for i in range(num_sents):
        ce_loss += masked_cross_entropy(text_logits[i], text_targets[i], target_lens[i], use_cuda=use_cuda)

    total_loss = ce_loss + Z_kl_mean + state_kl_mean
    
    print_iter_stats(iteration, total_loss, ce_loss, Z_kl_mean, state_kl_mean)
    
    return total_loss, ce_loss # tensor 

    


def print_iter_stats(iteration, total_loss, ce_loss, Z_kl, state_kl):

#    if iteration % args.log_every == 0 and iteration != 0:
    if True:
        print("Iteration: ", iteration) 
        print("Total: ", total_loss.cpu().data[0])
        print("CE: ", ce_loss.cpu().data[0])
        print("Z KL Div: ", Z_kl.cpu().data[0])
        print("State KL Div: ", state_kl.cpu().data[0])
        #print(reconstruct[:2, :5])



def check_save_model_path(save_model):
    save_model_path = os.path.abspath(save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


        
def train(args):
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

    if args.use_pretrained:
        pretrained = GloVe(name='6B', dim=args.emb_size, unk_init=torch.Tensor.normal_)
        vocab.load_vectors(pretrained)
        print("Vectors Loaded")

    print("Loading Dataset")
    dataset = du.RocStoryDataset(args.train_data, vocab) 
    print("Finished Loading Dataset {} examples".format(len(dataset)))

    sort_func = lambda x: max([len(x.sent_1),len(x.sent_2),len(x.sent_3),len(x.sent_4),len(x.sent_5)])

#    story_batches = du.RocStoryBatches(dataset, args.batch_size, sort_key=sort_func, train=True, sort_within_batch=True, device=-1)
    story_batches = du.RocStoryBatches(dataset, args.batch_size, train=True, device=-1)
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

    if args.load_model:
        print("Loading the Model")
        model = torch.load(args.load_model)
    else:
        print("Creating the Model")
        model = SLDS(args.hidden_size, args.emb_size, vocab, test_trans_matrix, use_cuda=use_cuda) #need to put arguments

    #create the optimizer
    if args.load_opt:
        print("Loading the optimizer state")
        optimizer = torch.load(args.load_opt)
    else:
        print("Creating the optimizer anew")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time() #start of epoch 1
    curr_epoch = 1
    valid_loss = [0.0]
    for iteration, story in enumerate(story_batches): #this will continue on forever (shuffling every epoch) till epochs finished
        batch, seq_lens = story_batches.combine_story(story) #should return batch tensor [num_sents, batch, seq_len] and seq_lens [num_sents, batch]

        if use_cuda:
            batch = batch.cuda()

        model.zero_grad()
        
        #Run the model
        #text_logits [num_sents, batch, seq_len, vocab size]
        #state_logits [num_sents, batch, num_classes]
        #Z_kl [batch]
        text_logits, state_logits, Z_kl, state_kl = model(batch, seq_lens)

        targets, target_lens = story_batches.convert_to_target(batch, seq_lens)

        if use_cuda:
            targets = targets.cuda()
            targen_lens= target_lens.cuda()

        loss, _ = compute_loss(text_logits, targets, target_lens, Z_kl, state_kl, iteration, use_cuda=use_cuda)
 
        # backward propagation
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # Optimize
        optimizer.step() 

        # End of an epoch - run validation
        if ((args.batch_size * iteration) % data_len == 0 or iteration % args.validate_after == 0) and iteration != 0:
            print("\nFinished Training Epoch/iteration {}/{}".format(curr_epoch, iteration))
            #PUT VALIDATION CODE IN HERE


            # Check max epochs and break
            if (args.batch_size * iteration) % data_len == 0:
                curr_epoch += 1
            if curr_epoch > args.epochs:
                print("Max epoch {}-{} reached. Exiting.\n".format(curr_epoch, args.epochs))
                break

        # Save the checkpoint
        if iteration % args.save_after == 0 and iteration != 0: 
            print("Saving checkpoint for epoch {} at {}.\n".format(curr_epoch, args.save_model))
            # curr_epoch and validation stats appended to the model name

            torch.save(model, args.save_model)
            torch.save(optimizer, "{}_{}".format("optimizer", args.save_model))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SLDS')
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--vocab', type=str, help='the vocabulary pickle file')
    parser.add_argument('--hidden_size', type=int, default=300, help='size of hidden state Z')
    parser.add_argument('--emb_size', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--layers', type=int, default=2, help='number of layers')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--log_every', type=int, default=200)
    parser.add_argument('--save_after', type=int, default=500)
    parser.add_argument('--validate_after', type=int, default=2500)
    parser.add_argument('--optimizer', type=str, default='adam', help='adam, adagrad, sgd')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed') 
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--bidir', type=bool, default=True, help='Use bidirectional encoder') 
    parser.add_argument('-src_seq_length', type=int, default=50, help="Maximum source sequence length")
    parser.add_argument('-max_decode_len', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('-save_model', default='model', help="""Model filename""")
    parser.add_argument('-latent_dim', type=int, default=256, help='The dimension of the latent embeddings')
    parser.add_argument('-use_pretrained', type=bool, default=False, help='Use pretrained glove vectors')
    parser.add_argument('-dropout', type=float, default=0.0, help='loss hyperparameters')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--load_opt', type=str)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open('{}_args.pkl'.format(args.save_model), 'wb') as fi:
        pickle.dump(args, fi)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # diff between train and classic: in classic pass .txt etension for files.
    train(args)



