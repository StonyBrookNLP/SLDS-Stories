#########
# code for S2S inference
########
import sys
sys.path.append('../')
import random
import numpy as np
import argparse
import torch
from utils import variable
import data_utils as du
from s2sa import S2SWithA
from torchtext.data import Iterator as BatchIter
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK

def generate(args):
    """
    Use the trained model for decoding
    Args
        args (argparse.ArgumentParser)
    """

    #Load the vocab
    inp_vocab = du.sentiment_label_vocab()
    out_vocab = du.load_vocab(args.out_vocab)
    eos_id = out_vocab.stoi[EOS_TOK]
    
    if args.test_data:
        # EXPLICITLY skip the header in val
        test_dataset = du.S2SSentenceDataset(args.test_data, inp_vocab, out_vocab, skip_header=True) 
        # batch_size is set to 1 rather than args.batch_size
        test_loader = BatchIter(test_dataset, 1, sort_key=lambda x:len(x.text), train=False, sort_within_batch=True, device=-1)  
        data_len = len(test_dataset)

    #Load the model 
    model = torch.load(args.load_model)

    model.eval()

    #sample_outputs(model, vocab)
    for batch_iter, batch in enumerate(test_loader): 
        input, input_lens = batch.text
        target, target_lens = batch.target
        input, input_lens, target = variable(input), variable(input_lens), variable(target)

        with torch.no_grad():
            outputs = model(input, input_lens, target, str_out=True, max_len_decode=target_lens.item())
    
        print("Input: {}".format(du.transform(input.squeeze().tolist(), inp_vocab.itos)))  
        print("TRUE: {}".format(du.transform(target.squeeze().tolist(), out_vocab.itos)))
        print("Decoded: {}\n\n".format(du.transform(outputs, out_vocab.itos)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='S2S')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--out_vocab', type=str, help='the output vocabulary pickle file', default='../data/rocstory_vocab_f5.pkl') 
    parser.add_argument('--seed', type=int, default=11, help='random seed') 
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--src_seq_len', type=int, default=50, help="Maximum source sequence length")
    parser.add_argument('--max_decode_len', type=int, default=50, help='Maximum prediction length.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    generate(args)



