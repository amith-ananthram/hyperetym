import os
import re
import glob
import math
import torch
import pickle
import numpy as np
import networkx as nx
from colour import Color

import drawSvg as draw
from drawSvg import Drawing
from hyperbolic import euclid, util
from hyperbolic.poincare.shapes import *
from hyperbolic.poincare import Transform

import loaders
from train_embeddings import Embeddings 
from manifold import PoincareManifold

from plot2 import plotPoincareDisc

MODEL_PATH = 'poincare-2'
START_NODE = ('rot', 'root')

if __name__ == '__main__':
    with open(os.path.join(MODEL_PATH, 'nodes.pkl'), 'rb') as f:
        nodes = pickle.load(f)
    with open(os.path.join(MODEL_PATH, 'edges.pkl'), 'rb') as f:
        edges = pickle.load(f)
    with open(os.path.join(MODEL_PATH, 'graph.pkl'), 'rb') as f:
        etym_wordnet = pickle.load(f)

    to_plot = {item:nodes[item] for item in (nx.descendants(etym_wordnet, START_NODE) | set([('rot', 'root'), (START_NODE)]))}

    torch.set_default_tensor_type(torch.DoubleTensor)
    for model_path in sorted(glob.glob(os.path.join(MODEL_PATH, 'model_checkpoint*.pt'))):
        epoch = int(re.search('model_checkpoint(\d*).pt', model_path).group(1))
        embeddings = Embeddings(nodes, PoincareManifold(), 2)
        embeddings.load_state_dict(torch.load(model_path, torch.device("cpu")))
        embeddings = embeddings.embeddings.weight.detach().numpy()
        embeddings = { item:embeddings[to_plot[item]] for item in to_plot }

        d = Drawing(2.1, 2.1, origin='center')
        d.draw(euclid.shapes.Circle(0, 0, 1), fill='silver')
        d.draw(Point(0, 0), radius=0.01, fill='black')

        points = []
        max_level = 0
        to_traverse = [(0, ('rot', 'root')), (1, START_NODE)] 
        while len(to_traverse) > 0:
            level, source = to_traverse.pop(0)

            max_level = max(max_level, level)
            points.append((level, Point(*embeddings[source]), source))

            if level == 0 and source == ('rot', 'root'):
                continue

            successors = etym_wordnet.successors(source)
            for successor in successors:
                to_traverse.append((level + 1, successor))
                # d.draw(Line.fromPoints(*embeddings[source], *embeddings[successor], segment=True), hwidth=radius/500)

        colors = list(Color("red").range_to(Color("green"), max_level + 1))
        for level, point, text in points:
            d.draw(point, radius=0.01, fill=colors[level].hex)   
            #d.draw(draw.Text('%s-%s' % text, 0.01, *point))      

        d.setRenderSize(w = 800)
        d.savePng('poincare-2-%s.png' % (str(epoch).zfill(2)))
