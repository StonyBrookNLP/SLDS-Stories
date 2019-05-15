import os
import random
import torch
import argparse
import pickle
import numpy as np
from torch import optim
from torch import nn
from torch.nn import functional as F
import torch.utils.data as tdata
import ../data_utils as du
from torchtext.data import Iterator as BatchIter
from model import S2SWithA
from utils import variable
from ../masked_cross_entropy import masked_cross_entropy


def get_data_loader(inp_vocab, out_vocab):
    """
    train and test loaders are each dict with key as the task_num. Total key length as num_task.
    """
       
    train_dataset = du.S2SSentenceDataset(args.train_data, inp_vocab, out_vocab, args.src_seq_len)                 
    train_loader = BatchIter(train_dataset, args.batch_size, sort_key=lambda x:len(x.text), train=True, sort_within_batch=True, device=-1)
 
    test_dataset = du.S2SSentenceDataset(args.test_data, args.test_data, inp_vocab, out_vocab, args.src_seq_len)
    # batch_size is set to 1 
    test_loader = BatchIter(test_dataset, 1, sort_key=lambda x:len(x.text), train=False, sort_within_batch=True, device=-1)
 
    return train_loader, test_loader


def test(model, test_loader):
    
    # set to eval mode
    model.eval()

    valid_loss = 0.0
    for batch_iter, batch in enumerate(test_loader): # should go over data only once

        #if batch_iter >= 10:
            #break

        input, input_lens = batch.text
        target, target_lens = batch.target 
        input, input_lens, target, target_lens = variable(input), variable(input_lens), variable(target), variable(target_lens)

        with torch.no_grad():
            logits = model(input, input_lens, target) # this max_len includes eos
 
        valid_loss += masked_cross_entropy(logits, target, target_lens)

    # set to train mode
    model.train()

    return valid_loss / len(test_loader.dataset)


def train(args):

    # Vocabulary will be created once for all the tasks. Must include tokens from all the tasks.
    print("Loading vocabulary.")
    inp_vocab = du.sentiment_label_vocab()
    out_vocab = du.load_vocab(args.out_vocab)
 
    print("Preparing the data loader.")
    train_loader, test_loader = get_data_loader(inp_vocab, out_vocab)

    print("Starting the learning process.")
    model = S2SWithA(args.emb_size, args.enc_hid_size, args.dec_hid_size, inp_vocab, out_vocab, layers=args.nlayers, use_cuda=args.cuda, bidir=args.bidir, dropout=args.dropout)

    # do cuda transfer before constructing the optimizer
    if torch.cuda.is_available() and args.cuda:
        print("Transferring the model to CUDA.")
        model.cuda()

    # set to train mode
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    c_epoch = 0
    train_loss = 0.0
    iters_per_epoch = int(np.ceil(len(train_loader.dataset) / float(args.batch_size)))
    for batch_iter, batch in enumerate(train_loader): # continues forever (shuffling every epoch) till args.epochs finished
 
        input, input_lens = batch.text
        target, target_lens = batch.target
        input, input_lens, target, target_lens = variable(input), variable(input_lens), variable(target), variable(target_lens)
        
        optimizer.zero_grad()

        logits = model(input, input_lens, target)
       
        loss = masked_cross_entropy(logits, target, target_lens)

        train_loss += loss.item()

        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
 
        # end of epoch / after intervals
        if (batch_iter+1) % iters_per_epoch ==  0 or (batch_iter + 1) % args.validate_after == 0:

            #avg_train_loss = train_loss / (batch_iter + 1)             
            #valid_loss = test(model, test_loader)
            #print("**Epoch {} iteration {} train_loss {:.4f} valid_loss {:.4f}**".format(c_epoch, batch_iter, avg_train_loss, valid_loss))            

            print("Saving checkpoint for epoch {}.\n".format(c_epoch))            
            torch.save(model, "{}_e{}_itr{}".format(args.expt_name, c_epoch, batch_iter))
            torch.save(optimizer, "{}_{}_e{}_itr{}".format("optimizer", args.expt_name, c_epoch, batch_iter))

            c_epoch += 1


        if c_epoch >= args.epochs:
            print("Max epoch {}/{} reached. Break\n".format(c_epoch, args.epochs[task]))
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='S2SA')  
    parser.add_argument('--train_data', type=str, help="Pass train files for all the tasks.")
    parser.add_argument('--test_data', type=str, help="Pass test files for all the tasks")
    parser.add_argument('--out_vocab', type=str, help='the output vocabulary pickle file')
    parser.add_argument('--emb_size', type=int, default=64, help='size of word embeddings')
    parser.add_argument('--enc_hid_size', type=int, default=512, help='size of encoder hidden')
    parser.add_argument('--dec_hid_size', type=int, default=512, help='size of encoder hidden')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer to be used.')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--bidir', type=bool, default=False, help='Use bidirectional encoder') 
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    #parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--validate_after', type=int, default=1500) 
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--seed', type=int, default=11, help='random seed') 
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--src_seq_length', type=int, default=50, help="Maximum source sequence length")
    parser.add_argument('--max_decode_len', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('--expt_name', type=str, default="new_expt", help='Parent folder under which all files will be created fo rthe expt..')
    
    args = parser.parse_args()
    print("\nAll args: {}\n".format(args))

    # Set all the seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)


    train(args)

