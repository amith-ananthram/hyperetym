import os
import argparse
import tqdm
import glob
import torch
import networkx as nx
import numpy as np

from functools import partial
from sklearn.metrics import average_precision_score

import loaders
from train_embeddings import Embeddings
from manifold import EuclideanManifold, PoincareManifold

from multiprocessing.pool import ThreadPool

def evaluate_reconstruction(nodes, etym_wordnet, model, node_split):
    prog = node_split[0]
    node_ixs = node_split[1]

    rank_off_tot = 0
    neighbors_tot = 0
    avg_precise_tot = 0
    labels = np.empty(model.embeddings.weight.size(0))
    for node in tqdm.tqdm(node_ixs) if prog else node_ixs:
        
        node = tuple(node)

        neighbors = np.array([nodes[n] for n in \
            (set(etym_wordnet.predecessors(node)) | \
                set(etym_wordnet.successors(node)))])
        neighbors_tot += neighbors.shape[0]
        
        #get all distances relative to target node (placeholder)
        #detach back to cpu and convert to numpy
        all_dists = model.manifold.distance(
            model.embeddings.weight[nodes[node]], model.embeddings.weight)
        all_dists = all_dists.detach().cpu().numpy()

        #set distance of target node to something large
        all_dists[nodes[node]] = 1e9

        #finding how many nodes rank ahead of our neighbors
        sorted_ix = all_dists.argsort()
        rank_off, = np.where(np.in1d(sorted_ix, neighbors))
        rank_off_tot += np.sum(rank_off) - (neighbors_tot * (neighbors_tot - 1) / 2)
        # rank_off_tot += np.sum(rank_off-np.arange(neighbors_tot))

        #reset to 0 and set only neighbors to 1, efficient way
        # labels = np.zeros(len(all_dists))
        labels.fill(0)
        labels[neighbors] = 1
 
        avg_precise_tot += average_precision_score(labels, -all_dists)

    return rank_off_tot, neighbors_tot, avg_precise_tot, len(node_ixs)
    
    # avg_rank_deviation = rank_off_tot / total_neighbors
    # avg_precision = avg_precise_tot / len(nodes.keys())
    #avg_rank_deviation, avg_precision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate embeddings')
    parser.add_argument('--dim', dest='dim')
    parser.add_argument('--manifold', dest='manifold')
    parser.add_argument('--model_dir', dest='model_dir')
    parser.add_argument('--bees', dest='bees', help='Num of parallel processes')
    parser.add_argument('--prog_one', action='store_true', dest='prog_one', default=False)
    args, unknown = parser.parse_known_args()

    # args.dim = 2
    # args.model_dir = 'embeddings/test-backup/'
    # args.manifold = 'poincare'
    # args.bees = 5

    num_bees = int(args.bees)

    #if num of bees > 10, force to only show 1 line
    if num_bees > 10:
        print("Because num of bees > 10, showing only 1 prog bar!")
        args.prog_one = True

    map_loc = None
    if not torch.cuda.is_available():
        map_loc = torch.device('cpu')

    # load data in graph form
    nodes, edges, etym_wordnet = loaders.get_etym_wordnet_dataset(langs=['eng'], decycle=False)

    if args.manifold == 'euclidean':
        manifold = EuclideanManifold()
    elif args.manifold == 'poincare':
        manifold = PoincareManifold() 
    else:
        raise Exception("Unsupported manifold: %s" % args.manifold)

    node_ixs = nodes.keys()

    for model_path in sorted(glob.glob(os.path.join(args.model_dir, 'model_checkpoint*.pt'))):
        print(model_path)

        embeddings = Embeddings(nodes, manifold, int(args.dim))
        embeddings.load_state_dict(torch.load(model_path, map_loc))

        if num_bees > 1:
            with ThreadPool(num_bees) as pool:
                f = partial(evaluate_reconstruction, \
                    nodes, etym_wordnet.etym_wordnet, embeddings)
                
                node_split = np.array_split(list(node_ixs), num_bees)
                if args.prog_one:
                    node_split = [(1,n) if i == 0 else (0,n) \
                        for i,n in zip(np.arange(len(node_split)), node_split)]
                else:
                    node_split = [(1,n) for n in node_split]

                output = pool.map(f, node_split)
                output = np.array(output).sum(axis=0).astype(float)
        else:
            output = evaluate_reconstruction(
                nodes, etym_wordnet.etym_wordnet, embeddings, (1,node_ixs))

        rank = output[0] / output[1]
        MAP = output[2] / output[3]
        print("rank=%s, map=%s" % (rank, MAP))


    # if args.bees > 1:
    #     with ThreadPool(args.bees) as pool:
    #         f = partial(
    #             evaluate_reconstruction, nodes, 
    #             etym_wordnet.etym_wordnet, embeddings)
    #         results = pool.map(f, np.array_split(objects, workers))
    #         results = np.array(results).sum(axis=0).astype(float)
    # else:
    #     results = reconstruction_worker(adj, model, objects, progress)
    # return float(results[0]) / results[1], float(results[2]) / results[3]

    
    #     rank, MAP = evaluate_reconstruction(
    #         nodes, etym_wordnet.etym_wordnet, embeddings)





