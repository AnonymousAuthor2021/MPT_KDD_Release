import argparse
import copy
import os
import time
import warnings

import dgl
import numpy as np
import psutil
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
import math
from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    LoadBalanceGraphDataset3,
    worker_init_fn,
)
from gcc.datasets.data_util import labeled_motif_batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear

import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, d_in, h_, d_out):
        super(Net, self).__init__()
        self.l1 = nn.Linear(d_in, h_)
        self.l2 = nn.Linear(h_, h_)
        self.l3=nn.Linear(h_, d_out)

    def forward(self,x):
        x = self.l1(x).clamp(min=0)
        x = self.l2(x).clamp(min=0)
        y_pred = self.l3(x)
        return y_pred


def parse_option():
    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb-freq", type=int, default=250, help="tb frequency")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch-size", type=int, default=32, help="batch_size")
    parser.add_argument("--num-workers", type=int, default=12, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=6, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=2000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")

    # optimization
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    # resume
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])

    parser.add_argument("--exp", type=str, default="")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index"] + GRAPH_CLASSIFICATION_DSETS)

    # model definition
    parser.add_argument("--model", type=str, default="gin", choices=["gat", "mpnn", "gin"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")

    # loss function
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)

    # random walk
    parser.add_argument("--rw-hops", type=int, default=256)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)

    # specify folder
    parser.add_argument("--model-path", type=str, default=None, help="path to save model")
    parser.add_argument("--tb-path", type=str, default=None, help="path to tensorboard")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    # finetune setting
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")

    # GPU setting
    parser.add_argument("--gpu", default=None, type=int, nargs='+', help="GPU id to use.")

    # cross validation
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--fold-idx", type=int, default=0, help="random seed.")
    parser.add_argument("--cv", action="store_true")
    # fmt: on

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def option_update(opt):
    opt.model_name = "{}_moco_{}_{}_{}_layer_{}_lr_{}_decay_{}_bsz_{}_hid_{}_samples_{}_nce_t_{}_nce_k_{}_rw_hops_{}_restart_prob_{}_aug_{}_ft_{}_deg_{}_pos_{}_momentum_{}".format(
        opt.exp, opt.moco, opt.dataset, opt.model, opt.num_layer,
        opt.learning_rate, opt.weight_decay, opt.batch_size, opt.hidden_size, opt.num_samples,
        opt.nce_t, opt.nce_k, opt.rw_hops, opt.restart_prob, opt.aug,
        opt.finetune, opt.degree_embedding_size, opt.positional_embedding_size, opt.alpha,)
    if opt.load_path is None:
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
    else:
        opt.model_folder = opt.load_path
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    return opt

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None))

def train_moco(epoch, model_name, train_loader, model, model_ema, contrast, criterion, optimizer, sw, opt, output_layer, output_layer_optimizer, global_output_layer, global_output_layer_optimizer):
    """
    one epoch training for moco
    """
    n_batch = train_loader.dataset.total // opt.batch_size
    model.train()
    model_ema.eval()
    print("pretrain")
    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_ema.apply(set_bn_train)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    global_loss_meter = AverageMeter()
    epoch_global_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    gnorm_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0
    end = time.time()

    # read global label
    graph_list = np.zeros(15)
    f = open("./motifs/" + model_name + "-counts.out")
    for line in f:
        nums = [int(x) for x in line.split()]
        graph_list += np.array(nums)
    global_label = torch.FloatTensor(np.array([x*1.0/sum(graph_list) for x in graph_list]))
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, label, degree_label = batch
        graph_q.to(torch.device(opt.gpu))       
        bsz = graph_q.batch_size
        # ===================Negative sampling forward=====================
        feat_q = model(graph_q)
        out = output_layer(feat_q)

        #global_feature.append(feat_q.detach().cpu())
        #mean_t = torch.mean(torch.cat(global_feature), dim=0, keepdim=True).squeeze()
        #print(len(global_feature), mean_t.shape)
        #continue
        #print(global_feature[0].shape)
        #print(global_feature, len(global_feature))
        #mean_t = mean_t.to(torch.device(opt.gpu)) 
        #global_out = global_output_layer(mean_t)
        degree_out = global_output_layer(feat_q)
        # print(feat_q.size(), feat_k.size())
        #print("negative sampling")
        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)

        # ===================backward=====================
        optimizer.zero_grad()
        loss = criterion(out, label)
        global_loss = criterion(degree_out, degree_label)
        output_layer_optimizer.zero_grad()
        loss = loss + global_loss
        loss.backward(retain_graph=True)
        global_output_layer_optimizer.zero_grad()
        global_loss.backward()

        torch.nn.utils.clip_grad_value_(output_layer.parameters(), 1)
        torch.nn.utils.clip_grad_value_(global_output_layer.parameters(), 1)
        grad_norm = clip_grad_norm(model.parameters(), opt.clip_norm)
        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        for param_group in output_layer_optimizer.param_groups:
            param_group["lr"] = lr_this_step
        for param_group in global_output_layer_optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()
        output_layer_optimizer.step()
        global_output_layer_optimizer.step()
        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        global_loss_meter.update(loss.item(), bsz)
        epoch_global_loss_meter.update(loss.item(), bsz)
        gnorm_meter.update(grad_norm, 1)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "prob {prob.val:.3f} ({prob.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch, idx + 1, n_batch,
                    batch_time=batch_time, data_time=data_time, loss=loss_meter,
                    prob=prob_meter, graph_size=graph_size, mem=mem.used / 1024 ** 3,))
        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("global_moco_loss", global_loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", prob_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, global_step)
            sw.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
            loss_meter.reset()
            global_loss_meter.reset()
            prob_meter.reset()
            graph_size.reset()
            gnorm_meter.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg, epoch_global_loss_meter.avg

def main(args):
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args = option_update(args)
    print(args)
    assert args.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args.gpu))
    assert args.positional_embedding_size % 2 == 0
    print("setting random seeds")
    mem = psutil.virtual_memory()
    print("before construct dataset", mem.used / 1024 ** 3)
    if args.dataset == "dgl":
        print(args)
        train_dataset = LoadBalanceGraphDataset3(
            args.model_path,
            dgl_graphs_file="./data_bin/dgl/"+args.model_path + ".bin",
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            num_copies=args.num_copies,
        )
    else:
        exit()
    mem = psutil.virtual_memory()
    print("before construct dataloader", mem.used / 1024 ** 3)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, collate_fn=labeled_motif_batcher(),
        shuffle=True if args.finetune else False, num_workers=args.num_workers, worker_init_fn=worker_init_fn,)
    mem = psutil.virtual_memory()
    print("before training", mem.used / 1024 ** 3)
    n_data = None
    model, model_ema = [
        GraphEncoder( positional_embedding_size=args.positional_embedding_size,
            max_node_freq=args.max_node_freq, max_edge_freq=args.max_edge_freq,
            max_degree=args.max_degree, freq_embedding_size=args.freq_embedding_size,
            degree_embedding_size=args.degree_embedding_size, output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size, edge_hidden_dim=args.hidden_size,
            num_layers=args.num_layer, num_step_set2set=args.set2set_iter,
            num_layer_set2set=args.set2set_lstm_layer, norm=args.norm, gnn_model=args.model,
            degree_input=True,)
        for _ in range(2)]

    # set the contrast memory and criterion
    contrast = MemoryMoCo(args.hidden_size, n_data, args.nce_k, args.nce_t, use_softmax=True).cuda(args.gpu)

    criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS()
    # criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS2()
    criterion = criterion.cuda(args.gpu)
    model = model.cuda(args.gpu)
    model_ema = model_ema.cuda(args.gpu)

    output_layer = nn.Linear(in_features=args.hidden_size, out_features=15)
    output_layer = output_layer.cuda(args.gpu)
    output_layer_optimizer = torch.optim.Adam(output_layer.parameters(),
        lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,)

    global_output_layer = Net(args.hidden_size, args.hidden_size, 1)
    global_output_layer = global_output_layer.cuda(args.gpu)
    global_output_layer_optimizer = torch.optim.Adam(global_output_layer.parameters(),
        lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,)
    def clear_bn(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.reset_running_stats()

    model.apply(clear_bn)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay,)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
            betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,)
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate,
            lr_decay=args.lr_decay_rate, weight_decay=args.weight_decay,)
    else:
        raise NotImplementedError

    # optionally resume from a checkpoint
    args.start_epoch = 1
    sw = SummaryWriter(args.tb_folder)
    losses = []
    global_losses = []
    early_stopping_round = 10
    
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")
        time1 = time.time()
        loss, global_loss = train_moco(epoch, args.model_path, train_loader, model, model_ema, contrast, criterion, optimizer, sw, args, output_layer, output_layer_optimizer, global_output_layer, global_output_layer_optimizer)
        if len(losses)> 100 and (losses[-10] - loss)/loss < 0:
            break
        losses.append(loss)
        global_losses.append(global_loss)
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
            print("==> Saving...")
            state = {"opt": args, "model": model.state_dict(), "contrast": contrast.state_dict(),
                "optimizer": optimizer.state_dict(), "epoch": epoch,}
            if args.moco:
                state["model_ema"] = model_ema.state_dict()
            save_file = os.path.join(args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch))
            torch.save(state, save_file)
            del state
        # saving the model
        print("==> Saving...")
        state = {"opt": args, "model": model.state_dict(), "contrast": contrast.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": epoch,}
        if args.moco:
            state["model_ema"] = model_ema.state_dict()
        save_file = os.path.join(args.model_folder, "current.pth")
        torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()
    print(losses, global_losses)
    plt.plot(list(range(len(losses))), losses, color='red', linestyle='--')
    plt.savefig(args.model_path + '.png',bbox_inches='tight')

if __name__ == "__main__":
    warnings.simplefilter("once", UserWarning)
    args = parse_option()
    args.gpu = args.gpu[0]
    main(args)
