import numpy as np
import torch
from torch import nn
from .data import get_batch 


# def batch_loss(logits: torch.Tensor, y: torch.Tensor):
#     r, n = logits.shape
#     zy = ndl.ops.summation(logits * nn.init.one_hot(n, y, device=logits.device),axes=1)
#     res = ndl.ops.summation(nn.ops.logsumexp(logits, (1,)) - zy)
#     return res

def accuracy(y_hat, y):
    y_hat = torch.argmax(y_hat, dim=1)
    return torch.sum(y_hat == y).item()

### PTB training ###
def epoch_general_ptb(data, model, seq_len, loss_fn=nn.CrossEntropyLoss(), opt=None,
        clip=None, device=None, dtype=None):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    if opt == None:
        model.eval()
    else:
        model.train()
    nbatch, batch_size = data.shape
    accum_loss = 0
    accum_acc = 0
    sum_samples = 0
    
    for i in range(0, nbatch - 1, seq_len):
        batch_x, batch_y = get_batch(data, i, seq_len, device=device, dtype=dtype)
        sum_samples += batch_y.shape[0]
        
        if opt == None:
            out, _ = model(batch_x)
            loss = loss_fn(out, batch_y)
        else:
            opt.zero_grad()
            out, _ = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            if getattr(opt, 'clip_grad_norm', None) is not None:
                if clip is not None:
                    opt.clip_grad_norm(clip)
                else:
                    opt.clip_grad_norm()
            opt.step()
        
        cur_batch_loss = loss.detach()
        cur_batch_succ = accuracy(out, batch_y)
        accum_loss +=  cur_batch_loss
        accum_acc += cur_batch_succ
        # if i%100==0:
        #     print("done:[{}], left:[{}], total:[{}]".format(i, nbatch-i, nbatch))
        #     print("batch:{} \t batch_loss:[{}] \t batch_acc:[{}]".format(i, cur_batch_loss, cur_batch_succ))
        #     print()
    return accum_acc*(1.0/sum_samples), accum_loss * (1.0/sum_samples)  


def train_ptb(model, data, seq_len, n_epochs, optimizer, 
          lr, weight_decay, loss_fn, clip=None,
          epoch_func = epoch_general_ptb,
          device=None, dtype=None):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        print("epoch:{}".format(i))
        avg_acc, avg_loss = epoch_func(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=opt, clip=clip, device=device, dtype=dtype)
        print("loss: ", avg_loss, "acc: ", avg_acc)
    return avg_acc, avg_loss


def evaluate_ptb(model, data, seq_len, loss_fn=nn.CrossEntropyLoss(),
        epoch_func = epoch_general_ptb,
        device=None, dtype=None):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    avg_acc, avg_loss = epoch_func(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=None, device=device, dtype=dtype)
    return avg_acc, avg_loss