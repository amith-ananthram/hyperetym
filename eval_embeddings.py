import argparse
import tqdm
import torch
import networkx as nx
import numpy as np
import loaders
import manifold

from multiprocessing.pool import ThreadPool
from functools import partial
from sklearn.metrics import average_precision_score

def evaluate_reconstruction(etym_wordnet, model, prog=True):

    graph = etym_wordnet.etym_wordnet
    nodes = etym_wordnet.nodes
    node_list = list(nodes.values())

    rank_off_tot = 0
    avg_precise_tot = 0
    labels = np.empty(model.lt.weight.size(0))
    for node in tqdm(node_list) if prog else node_list:
        
        neighbors = np.array([nodes[n] for n in graph.neighbors(node)])
        total_neighbors = neighbors.shape[0]
        
        #get all distances relative to target node (placeholder)
        #detach back to cpu and convert to numpy
        all_dists = model.distance(model.lt.weight[node], model.lt.weight)
        all_dists = all_dists.detach().cpu().numpy()

        #set distance of target node to something large
        all_dists[nodes[node]] = 1e9

        #finding how many nodes rank ahead of our neighbors
        sorted_ix = all_dists.argsort()
        rank_off, = np.where(np.in1d(sorted_ix, neighbors))
        rank_off_tot += np.sum(rank_off+1-np.arange(total_neighbors))

        #reset to 0 and set only neighbors to 1, efficient way
        labels = np.fill(0)
        labels[neighbors] = 1
 
        avg_precise_tot += average_precision_score(labels, -all_dists)

    avg_rank_deviation = rank_off_tot / total_neighbors
    avg_precision = avg_precise_tot / len(node_list)

    return avg_rank_deviation, avg_precision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate embeddings')
    parser.add_argument('--progBar', action='store_true', dest='progBar', default=True)
    args, unknown = parser.parse_known_args()

    # load data in graph form
    nodes, edges, etym_wordnet = loaders.get_etym_wordnet_dataset()

    # model placeholder
    model = manifold.EuclideanManifold

    Rank, MAP = evaluate_reconstruction(
        etym_wordnet, model, args.progBar)






