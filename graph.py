"""
Markov Chain Graph Representation


"""
# class Edge(object):
#     """ Representation for a weighted, directed edge
#
#     """
#
#     def __init__(self, tail, head):
#         """ Creates a weighted, directed path between two nodes
#
#         :param tail: origin node
#         :param head: destination node
#         :param weight: weight of path
#
#         """
#         self.node_tail = tail
#         self.node_head = head
#

class Node(object):
    """ Representation for a node in the graph

    """
    def __init__(self, name):
        self.name = name
        # Holds a set of all the edges this node is a tail of
        self.paths = {}
        self.edge_count = 0.0

    def add_path(self, edge):
        if edge.name in self.paths:
            self.paths[edge.name] += 1
        else:
            self.paths[edge.name] = 1
        self.edge_count += 1.0



class MarkovChain(object):
    """Graph representation of the Markov Chain

    Directed and weighted graph with nodes for each token.

    """

    def __init__(self):
        """ Creates an empty graph

        nodes holds a set of all the nodes in the graph

        paths holds a Dict of all the paths in the graph

        """
        self.nodes = {}

    def add_node(self, node):
        """ Adds a node to our graph


        """
        if node in self.nodes:
            return
        self.nodes[node.name] = node

    def update_node(self, node):
        """

        Updates the paths of a current node
        """
        self.nodes[node.name].paths = node.paths
        self.nodes[node.name].edge_count = node.edge_count




