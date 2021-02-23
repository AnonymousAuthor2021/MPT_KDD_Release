import argparse
import networkx as nx
import os
import dgl
import numpy as np
import torch
from gcc.models import GraphEncoder

class LoadBalanceGraphDataset3(torch.utils.data.IterableDataset):
    def __init__(self, name, dgl_graphs_file, rw_hops=64, restart_prob=0.8,
        positional_embedding_size=32, step_dist=[1.0, 0.0, 0.0],
        num_workers=1,
        num_samples=10000, num_copies=1,
        graph_transform=None, aug="rwr", num_neighbors=5, ):
        super(LoadBalanceGraphDataset3).__init__()
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        self.dgl_graphs_file = dgl_graphs_file
        print(dgl.data.utils.load_labels(dgl_graphs_file))
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)[
            "graph_sizes"
        ].tolist()
        print(dgl_graphs_file, "load graph done1")
        #print(dgl.data.utils.load_labels(dgl_graphs_file))
        assert num_workers % num_copies == 0
        jobs = [list() for i in range(num_workers // num_copies)]
        workloads = [0] * (num_workers // num_copies)
        graph_sizes = sorted(
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        )
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.jobs = jobs * num_copies
        self.total = self.num_samples * num_workers
        self.graph_transform = graph_transform
        assert aug in ("rwr", "ns")
        self.aug = aug
        print("start load motif")
        self.motif_dict, self.graph_motif_list = load_motif(name)
        print(num_samples, num_workers)


    def __len__(self):
        return self.num_samples * num_workers

    def __iter__(self):
        #print(self.length)
        degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob.numpy()
        )
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        #print(self.length, idx)
        graph_idx = 0
        node_idx = idx
        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        other_node_idx = node_idx
        max_nodes_per_seed = max(
            self.rw_hops,int(
            ((self.graphs[graph_idx].in_degree(node_idx) ** 0.75)* math.e/ (math.e - 1)/ self.restart_prob)
        + 0.5),)
        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx], seeds=[node_idx, other_node_idx],
            restart_prob=self.restart_prob, max_nodes_per_seed=max_nodes_per_seed,)
        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx], graph_idx=graph_idx, seed=node_idx,
            trace=traces[0], positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
        #print(idx)
        #print(idx, traces[0], len(graph_q.nodes()))
        return graph_q, self.motif_dict[(graph_idx, node_idx)]

def test_moco(train_loader, model, opt):
    print(opt.norm)
    model.eval()
    emb_list = []
    for idx, batch in enumerate(train_loader):
        if opt.norm:
            graph_q, graph_k = batch
            bsz = graph_q.batch_size
            graph_q.to(opt.device)
            graph_k.to(opt.device)
            
            with torch.no_grad():#not necessary 
                feat_q = model(graph_q)
                feat_k = model(graph_k)
            #print(feat_q.shape, feat_k.shape, (feat_q + feat_k).shape)
            assert feat_q.shape == (bsz, opt.hidden_size)
            emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
        else:
            graph_q, _ = batch
            bsz = graph_q.batch_size
            #print(len(graph_q.nodes()))
            graph_q.to(opt.device)
            with torch.no_grad():
                feat_q = model(graph_q)
            assert feat_q.shape == (bsz, opt.hidden_size)
            emb_list.append(feat_q.detach().cpu())
    return torch.cat(emb_list)

def main(args_test):
    data = args_test.dataset
    g = nx.read_edgelist("../../data/" + "processed/" + data + ".edges")
    graph = dgl.DGLGraph(g)
    print(graph)
    for node_idx in range(len(graph.nodes)):
        print(node_idx)
        max_nodes_per_seed = max(self.rw_hops,int(((self.graphs[graph_idx].in_degree(node_idx) ** 0.75)* math.e/ (math.e - 1)/ self.restart_prob)+ 0.5),)
        #traces = dgl.contrib.sampling.random_walk_with_restart(
        #    self.graphs[graph_idx], seeds=[node_idx, other_node_idx],
        #    restart_prob=self.restart_prob, max_nodes_per_seed=max_nodes_per_seed,)
        #graph_q = data_util._rwr_trace_to_dgl_graph(
        #    g=self.graphs[graph_idx], graph_idx=graph_idx, seed=node_idx,
        #    trace=traces[0], positional_embedding_size=self.positional_embedding_size,
        #)
    exit(0)
    args.device = torch.device("cpu") if args.gpu is None else torch.device(args.gpu)
    train_dataset = NodeClassificationDataset(
        dataset=args_test.dataset, rw_hops=args.rw_hops, subgraph_size=args.subgraph_size,
        restart_prob=args.restart_prob, positional_embedding_size=args.positional_embedding_size,)
    args.batch_size = len(train_dataset) # not necessary
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size,
        collate_fn=batcher(), shuffle=False, num_workers=args.num_workers,)
    # create model and optimizer
    model = GraphEncoder(
        positional_embedding_size=args.positional_embedding_size,
        max_node_freq=args.max_node_freq, max_edge_freq=args.max_edge_freq,
        max_degree=args.max_degree, freq_embedding_size=args.freq_embedding_size,
        degree_embedding_size=args.degree_embedding_size, output_dim=args.hidden_size,
        node_hidden_dim=args.hidden_size, edge_hidden_dim=args.hidden_size,
        num_layers=args.num_layer, num_step_set2set=args.set2set_iter, num_layer_set2set=args.set2set_lstm_layer,
        gnn_model=args.model, norm=args.norm, degree_input=True,)
    model = model.to(args.device)

    model.load_state_dict(checkpoint["model"])
    del checkpoint
    emb = test_moco(train_loader, model, args)
    print(os.path.join(args_test.save_path, args_test.dataset))
    np.save(os.path.join(args_test.save_path, args_test.dataset), emb.numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    # fmt: off
    parser.add_argument("--dataset", type=str, default="dgl")
    # fmt: on
    main(parser.parse_args())