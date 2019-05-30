import torch

def accuracy(preds,target):
#     labels = target.argmax(dim=1)
    acc = preds.argmax(dim=1).eq(target.argmax(dim=1)).float().mean()
    return acc
