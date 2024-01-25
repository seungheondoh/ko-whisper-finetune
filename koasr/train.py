import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import whisper
from koasr.data.ko_speech import MelDataset, WhisperDataCollatorWhithPadding
from koasr.finetuner import FinetuneModel
from koasr.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams
from koasr.utils.eval_utils import print_model_params

parser = argparse.ArgumentParser(description='')
parser.add_argument('--framework', type=str, default="pretrain")
parser.add_argument('--model_name', type=str, default="small")
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--total_steps', default=2**14, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_steps', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=10, type=int)

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)

def main_worker(args):
    options = whisper.DecodingOptions(language="ko", without_timestamps=True)
    tokenizer = whisper.tokenizer.get_tokenizer(True, language="ko", task=options.task)
    collate_fn = WhisperDataCollatorWhithPadding()
    train_dataset = MelDataset(split="train", tokenizer=tokenizer)
    test_dataset = MelDataset(split="test", tokenizer=tokenizer)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    ft_model = FinetuneModel(
        model_name = args.model_name,
        tokenizer = tokenizer
    )
    print_model_params(ft_model.model.encoder)
    print_model_params(ft_model.model.decoder)

    torch.cuda.set_device(args.gpu)
    ft_model = ft_model.cuda(args.gpu)
    optimizer = torch.optim.AdamW(ft_model.parameters(), args.lr)
    save_dir = f"exp/{args.model_name}"
    args.epochs = args.total_steps // len(train_loader)
    args.start_epoch = args.start_steps // len(train_loader)
    args.warmup_steps = 5000
    
    logger = Logger(save_dir)
    save_hparams(args, save_dir)
    print(args.start_epoch, args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, ft_model, optimizer, epoch, logger, args)

    torch.save({'epoch': epoch, 'state_dict': ft_model.model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/last.pth')
    print("We are at epoch:", epoch)

def train(train_loader, model, optimizer, epoch, logger, args):
    train_losses = AverageMeter('Train Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),[train_losses],prefix="Epoch: [{}]".format(epoch))
    iters_per_epoch = len(train_loader)
    model.train()
    for data_iter_step, batch in enumerate(train_loader):
        current_step = int(epoch * iters_per_epoch) + data_iter_step
        lr = adjust_learning_rate(optimizer, current_step, args)
        if args.gpu is not None:
            _batch = {
                "mels": batch["mels"].cuda(args.gpu, non_blocking=True),
                "labels": batch["labels"].cuda(args.gpu, non_blocking=True),
                "tokens": batch["tokens"].cuda(args.gpu, non_blocking=True)
            }
        optimizer.zero_grad()
        # compute output
        loss = model.training_step(_batch)
        train_losses.step(loss.item(), args.batch_size)
        logger.log_train_loss(loss, epoch * iters_per_epoch + data_iter_step)
        logger.log_learning_rate(lr, epoch * iters_per_epoch + data_iter_step)
        loss.backward()
        optimizer.step()
        if data_iter_step % args.print_freq == 0:
            progress.display(data_iter_step)

def adjust_learning_rate(optimizer, current_step, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_steps = args.warmup_steps
    total_steps = args.total_steps
    if current_step < warmup_steps:
        lr = args.lr * current_step / warmup_steps
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()

    