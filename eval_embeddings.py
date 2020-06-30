import argparse
import tqdm
import glob
import torch
import networkx as nx
import numpy as np
import loaders
import manifold

from functools import partial
from sklearn.metrics import average_precision_score

def evaluate_reconstruction(nodes, etym_wordnet, model, prog=True):
    rank_off_tot = 0
    avg_precise_tot = 0
    labels = np.empty(model.embeddings.weight.size(0))
    for node in tqdm(nodes.keys()) if prog else node_list:
        neighbors = np.array([nodes[n] for n in graph.neighbors(node)])
        total_neighbors = neighbors.shape[0]
        
        #get all distances relative to target node (placeholder)
        #detach back to cpu and convert to numpy
        all_dists = model.manifold.distance(model.embeddings.weight[node], model.embeddings.weight)
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
    avg_precision = avg_precise_tot / len(nodes.keys())

    return avg_rank_deviation, avg_precision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate embeddings')
    parser.add_arugment('--dim', dest='dim')
    parser.add_argument('--manifold', dest='manifold')
    parser.add_argument('--model_dir', dest='model_dir')
    parser.add_argument('--prog_bar', action='store_true', dest='prog_bar', default=True)
    args, unknown = parser.parse_known_args()

    # load data in graph form
    nodes, edges, etym_wordnet = loaders.get_etym_wordnet_dataset(langs=['eng'])

    if manifold == 'euclidean':
        manifold = EuclideanManifold()
    elif manifold == 'poincare':
        manifold = PoincareManifold() 
    else:
        raise Exception("Unsupported manifold: %s" % manifold)

    for model_path in sorted(glob.glob(os.path.join(args.model_dir, 'model_checkpoint*.pt'))):
        print(model_path)

        embeddings = Embeddings(nodes, manifold, int(args.dimdim))
        embeddings.load_state_dict(torch.load(model_path))

        rank, MAP = evaluate_reconstruction(
            etym_wordnet, embeddings, args.prog_bar)
        print("rank=%s, map=%s" % (rank, MAP))






