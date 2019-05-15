import os
import torch
from torch.autograd import Variable

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


def check_path_no_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
