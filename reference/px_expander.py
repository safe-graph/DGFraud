'''
Expander code for clipping individual gradients.

This code is due to Mikko Heikkilä (@mixheikk)
'''
import torch
from torch.autograd import Variable

import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# clip and accumulate clipped gradients
def acc_scaled_grads(model, C, cum_grads):
    # this two 'batch size' should be equal.
    assert model.batch_size == model.batch_proc_size
    batch_size = model.batch_proc_size

    g_norm = Variable(torch.zeros(batch_size), requires_grad=False).to(device)

    counter1 = 0
    counter2 = 0
    g_norm = {}
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        if len(p.data.shape) == 2:
            continue
        counter2 += 1
        if p.grad is not None:
            counter1 += 1
            g_norm[str(counter2)] = torch.sqrt(torch.sum(p.grad.view(p.shape[0], -1) ** 2, 1))
    # print for debug:
    # print(counter1)
    # print(counter2)

    # do clipping and accumulate
    for p, key in zip(filter(lambda p: p.requires_grad, model.parameters()), cum_grads.keys()):
        if len(p.data.shape) == 2:
            continue
        if p is not None:
            cum_grads[key] = torch.sum((p.grad / torch.clamp(g_norm[key].contiguous().view(-1, 1, 1) / C, min=1)), dim=0)


# add noise and replace model grads with cumulative grads
def add_noise_with_cum_grads(model, C, sigma, cum_grads, samp_num):
    for p, key in zip(filter(lambda p: p.requires_grad, model.parameters()), cum_grads.keys()):
        if len(p.data.shape) == 2:
            continue
        proc_size = model.batch_size
        if key == '1':
            proc_size = proc_size * (samp_num+1)
        if p.grad is not None:
            # add noise to summed clipped pars
            if proc_size > 1:
                p.grad = ((cum_grads[key].expand(proc_size, -1, -1) +
                           Variable(
                               (sigma * C)*torch.normal(mean=torch.zeros_like(p.grad[0]).data, std=1.0).expand(proc_size, -1, -1)
                                )
                           ) / proc_size
                          ).to(device)
            # p.grad = (torch.sum((p.grad), dim=0).expand(proc_size, -1, -1)) / proc_size