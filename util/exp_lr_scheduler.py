import torch
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_decay ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr