import sys

from model import *
import data_utils as du
from torchtext.data import Iterator as BatchIter
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK

hidden_size = 512
embed_size = 300 
learning_rate = 0.0001
n_epochs = 100000
grad_clip = 1.0

kld_start_inc = 10000
kld_weight = 0.0
kld_max = 1.0
kld_inc = 0.000002

x_0 = 10000.0 #Where to stop annaeling kl term

# Training
# ------------------------------------------------------------------------------

print("\nLoading Vocab")
vocab = du.load_vocab(sys.argv[2])
vocab_len = len(vocab.stoi.keys())
print("Vocab Loaded, Size {}".format(len(vocab.stoi.keys())))

dataset = du.SentenceDataset(sys.argv[1], vocab, 50, add_eos=False) #put in filter pred later
print("Finished Loading Dataset {} examples".format(len(dataset)))
batches = BatchIter(dataset, 1, sort_key=lambda x:len(x.text), train=True, sort_within_batch=True, device=-1)
data_len = len(dataset)


e = EncoderRNN(vocab_len ,hidden_size, embed_size)
d = DecoderRNN(embed_size, hidden_size, vocab_len)
vae = VAE(e, d)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

if USE_CUDA:
    vae.cuda()
    criterion.cuda()

log_every = 20
save_every = 5000


def save():
    save_filename = 'vae.pt'
    torch.save(vae, save_filename)
    print('Saved as %s' % save_filename)

try:
    #for epoch in range(n_epochs):
    for epoch, bl in enumerate(batches): #this will continue on forever (shuffling every epoch) till epochs finished
        batch, batch_lens = bl.text
        target, target_lens = bl.target 

        target = Variable(target.squeeze(0).cuda())

        batch = Variable(batch.squeeze(0).cuda())


        optimizer.zero_grad()

        m, l, z, decoded = vae(batch)

        loss = criterion(decoded, target)

        KLD = (-0.5 * torch.sum(l - torch.pow(m, 2) - torch.exp(l) + 1, 1)).mean().squeeze()
        loss += KLD * kld_weight

        kld_weight = min(1.0, epoch / x_0)

        loss.backward()
        # print('from', next(vae.parameters()).grad.data[0][0])
        ec = torch.nn.utils.clip_grad_norm(vae.parameters(), grad_clip)
        # print('to  ', next(vae.parameters()).grad.data[0][0])
        optimizer.step()

        if epoch % log_every == 0:
            print('[%d] %.4f %.4f' % (epoch, loss.data[0], kld_weight))

        if epoch > 0 and epoch % save_every == 0:
            save()

    save()

except KeyboardInterrupt as err:
    print("ERROR", err)
    print("Saving before quit...")
    save()

