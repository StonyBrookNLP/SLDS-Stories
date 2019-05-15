#########
# WIP
########


def generate(args):
    """
    Use the trained model for decoding
    Args
        args (argparse.ArgumentParser)
    """

    #Load the vocab
    vocab = du.load_vocab(args.vocab)
    eos_id = vocab.stoi[EOS_TOK]
    pad_id = vocab.stoi[PAD_TOK]
 
    
    test_dataset = du.SentenceDataset(args.test_data[task], inp_vocab, out_vocab, args.src_seq_len, add_eos=True) 
        # batch_size is set to 1 rather than args.batch_size
    test_loader[task] = BatchIter(test_dataset, 1, sort_key=lambda x:len(x.text), train=False, sort_within_batch=True, device=-1)  
    data_len = len(dataset)

    #Load the model 
    model = torch.load_state_dict(torch.load(args.load_model))

    model.eval()

    #sample_outputs(model, vocab)
    decode(args, model, batches, vocab)


def decode(args, model, batches, vocab):

    for batch_iter, bl in enumerate(batches):
        input, input_lens = bl.text
        target, target_lens = bl.target 
        batch = variable(input, volatile=True)
       
        with torch.no_grad():
            outputs = model(input, input_lens, str_out=True, max_len_decode=args.max_len_decode)

        print("TRUE: {}".format(transform(batch.data.squeeze(), vocab.itos)))
        print("Decoded: {}\n\n".format(transform(outputs, vocab.itos)))


def transform(output, dict):
    out = ""
    for i in output:
        out += " " + dict[i]
    return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DAVAE')

    args = parser.parse_args()
    parser.add_argument('--load_model', type=str)


    parser.add_argument('--test_data', type=str, nargs='+', help="Pass test files for all the tasks")
    parser.add_argument('--inp_vocab', type=str, help='the input vocabulary pickle file')
    parser.add_argument('--out_vocab', type=str, help='the output vocabulary pickle file')
    parser.add_argument('--sample_size', type=int, default=200, help='sample size')
    parser.add_argument('--num_task', type=int, default=2, help='number of tasks to learn')

    parser.add_argument('--emb_size', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--enc_hid_size', type=int, default=512, help='size of encoder hidden')
    parser.add_argument('--dec_hid_size', type=int, default=512, help='size of encoder hidden')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed') 
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--ewc', action='store_true', help='use EWC penalty for subsequent tasks')
    parser.add_argument('--bidir', type=bool, default=False, help='Use bidirectional encoder') 
    parser.add_argument('--src_seq_len', type=int, default=50, help="Maximum source sequence length")
    parser.add_argument('--max_decode_len', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('--save_model', default='model', help="""Model filename""") 
    parser.add_argument('--use_pretrained', type=bool, default=True, help='Use pretrained glove vectors') 




    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    generate()



