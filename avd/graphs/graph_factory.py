import random

from tqdm import tqdm

import avd.utils.utils as utils
from avd.graphs.gtgraph import GtGraph
from avd.graphs.nxgraph import NxGraph
from avd.samplers.graph_sampler import GraphSampler


def get_graph(package="Networkx"):
    return NxGraph


class GraphFactory(object):
    def factory(self, graph_config, labels=None, fake_users_number=None, limit=5000000,
                package="Networkx"):
        if not labels:
            labels = {"neg": "Real", "pos": "Fake"}
        if graph_config.type == "regular":
            return self.make_graph(graph_config.data_path, graph_config.is_directed, graph_config.labels_path,
                                   max_num_of_edges=limit, start_line=graph_config.first_line,
                                   package=package, pos_label=labels["pos"],
                                   neg_label=labels["neg"], delimiter=graph_config.delimiter)

    def make_graph(self, graph_path, is_directed=False, labels_path=None, package="Networkx", pos_label=None,
                   neg_label=None, start_line=0, max_num_of_edges=10000, weight_field=None, blacklist_path=False,
                   delimiter=','):
        """
            Loads graph into specified package.
            Parameters
            ----------
            blacklist_path
            delimiter
            graph_path : string

            is_directed : boolean, optional (default=False)
               Hold true if the graph is directed otherwise false.

            labels_path : string or None, optional (default=False)
               The path of the node labels file.

            package : string(Networkx, GraphLab or GraphTool), optional (default="Networkx")
               The name of the package to should be used to load the graph.

            pos_label : string or None, optional (default=None)
               The positive label.

            neg_label : string or None, optional (default=None)
               The negative label.

            start_line : integer, optional (default=0)
               The number of the first line in the file to be read.

            max_num_of_edges : integer, optional (default=10000000)
               The maximal number of edges that should be loaded.

            weight_field : string

            Returns
            -------
            g : AbstractGraph
                A graph object with the randomly generated nodes.

        """
        graph = get_graph(package)(is_directed, weight_field)
        if labels_path:
            print("Loading labels...")
            graph.load_labels(labels_path)
        if blacklist_path:
            print("Loading black list...")
            blacklist = utils.read_set_from_file(blacklist_path)
        else:
            blacklist = []
        graph.map_labels(positive=pos_label, negative=neg_label)
        print("Loading graph...")
        graph.load_graph(graph_path, start_line=start_line, limit=max_num_of_edges, blacklist=blacklist,
                         delimiter=delimiter)
        print("Data loaded.")
        return graph

    def load_saved_graph(self, graph_path, is_directed=False, labels_path=False, package="Networkx", pos_label=None,
                         neg_label=None, weight_field=None):
        """
            Load graph that was save by the library into specified package.
            Parameters
            ----------
            graph_path : string

            is_directed : boolean, optional (default=False)
               Hold true if the graph is directed otherwise false.

            labels_path : string or None, optional (default=False)
               The path of the node labels file.

            package : string(Networkx, GraphLab or GraphTool), optional (default="Networkx")
               The name of the package to should be used to load the graph.

            pos_label : string or None, optional (default=None)
               The positive label.

            neg_label : string or None, optional (default=None)
               The negative label.

            weight_field : string

            Returns
            -------
            g : AbstractGraph
                A graph object with the randomly generated nodes.

        """
        graph = get_graph(package)(is_directed, weight_field)
        if labels_path and utils.is_valid_path(labels_path):
            print("Loading labels...")
            graph.load_labels(labels_path)
        graph.map_labels(positive=pos_label, negative=neg_label)
        print("Loading graph...")
        graph = graph.load_saved_graph(graph_path)
        print("Data loaded.")
        return graph

