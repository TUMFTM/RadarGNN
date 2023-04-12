
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List

from gnnradarobjectdetection.graph_constructor.features import get_En_equivariant_point_pair_metrics


class Graph():
    """ General graph.

    Attributes:
        X_feat: Node feature matrix
        E_feat: Edge feature matrix
        A: Adjacency Matrix
        E: Edge connection matrix (Alternative representation of sparse adjacency matrix)

    Methods:
        build: Builds the graph.
        add_node_features: Add a node feature to X_feat.
        get_degree: Get the degree of each node of the graph.
        show: Plot the graph.
    """
    def __init__(self):
        self.X_feat = None  # all node features
        self.E_feat = None  # all edge features
        self.A = None       # adjacency matrix
        self.E = None       # edge connection matrix (alternative for A)

    def build(self, X: np.ndarray, routine: str, k: int = 6, r: float = 1) -> None:
        """ Builds a graph.

        Creates edges between the given points X and stores them as adjacency matrices A and E.
        All dimensions of X are used for distance calculation.

        Args:
            X: Input point cloud for which edges are created.
            routine: Graph constructing algorithm to use.
            k (optional): Parameter k for KNN graph.
            r (optional): Radius in radius neighborhood graph.
        """

        if X.shape[0] > 1:
            if (routine == "knn"):
                self.__build_knn(X, k)

            elif (routine == "radius"):
                self.__build_rn(X, r)

    def __build_knn(self, X: np.ndarray, k: int) -> None:
        """ Builds k-nn graph.

        Defines edges by knn algorithm for n-dimensional features X.
        """
        A_sparse = kneighbors_graph(
            X, k, mode='connectivity', include_self=False)
        A = A_sparse.toarray()

        e1 = A_sparse.nonzero()[0].reshape(-1, 1)
        e2 = A_sparse.nonzero()[1].reshape(-1, 1)
        E = np.concatenate((e1, e2), axis=1)

        self.A = A
        self.E = E

    def __build_rn(self, X: np.ndarray, r: float) -> None:
        """ Builds radius neighbor graph.

        Defines edges in a radius for n-dimensional features X
        """
        A_sparse = radius_neighbors_graph(
            X, r, mode='connectivity', include_self=False)
        A = A_sparse.toarray()

        e1 = A_sparse.nonzero()[0].reshape(-1, 1)
        e2 = A_sparse.nonzero()[1].reshape(-1, 1)
        E = np.concatenate((e1, e2), axis=1)

        self.A = A
        self.E = E

    def add_node_features(self, feat: np.ndarray) -> None:
        if self.X_feat is None:
            self.X_feat = feat
        else:
            if feat.shape[0] == self.X_feat.shape[0]:
                self.X_feat = np.concatenate((self.X_feat, feat), axis=1)
            else:
                raise Exception("Feature dimension not compatible")

    def get_degree(self) -> list:
        G = nx.from_numpy_matrix(self.A)
        degrees = [val for (node, val) in G.degree()]
        return degrees

    def show(self, node_size: float = 60) -> None:
        G = nx.from_numpy_matrix(self.A)
        fig, ax = plt.subplots()
        nx.draw(G, ax=ax, node_size=node_size)


class GeometricGraph(Graph):
    """ Geometric graph that deviates between spatial and non spatial features.

    This graph is tailored to radar data with different spatial features.

    Attributes:
        X: Spatial coordinates of each node.
        V: Velocity vector of each node.
        F: Non-spatial features of each node.

    Methods:
        add_invariant_feature: Add features to F.
        add_degree_to_inv_features: Add the degree of each node as additional feature to F.
        extract_node_pair_features: Extract features of each point-pair and store them in E_feat.
        extract_single_node_features: Extract features of each point and store them in X_feat.
        show: Visualize the graph.
    """

    def __init__(self):
        super().__init__()
        self.X = None       # node features with spatial coordinates
        self.V = None       # node features with velocity
        self.F = None       # additional E(3) invariant features (e.g. rcs, timestamp, degree) stored as dict

    def add_invariant_feature(self, name: str, F_add: np.ndarray) -> None:
        if self.F is None:
            self.F = {name: F_add}
        else:
            self.F[name] = F_add

    def add_degree_to_inv_features(self) -> None:
        deg = self.get_degree()
        deg_arr = np.array([deg]).reshape(len(deg), 1)
        self.add_invariant_feature("degree", deg_arr)

    def extract_node_pair_features(self, features: List[str], edge_mode: str) -> None:
        """ Extracts features of neighboring nodes.

        Saves the features of each edge in the edge feature matrix E_feat.

        Args:
            features: Specifies which features to extract.
                - point_pair_features             SE(n) invariant
                - spatial_euclidean_distance      SE(n) invariant
                - velocity_euclidean_distance     SE(n) invariant
                - relative_position               T(n) invariant
                - relative_velocity               T(n) invariant

            edge_mode: Specifies if feat(x_i, x_j) == feat(x_j, x_i) or not.
                - undirected: Edge direction is not considered: feat(x_i, x_j) == feat(x_j, x_i)
                - directed: Edge direction is considered: feat(x_i, x_j) = feat(x_j, x_i)
        """

        # get total number of features to extract
        num_edges = self.E.shape[0]
        num_feat = 0
        for feature in features:
            if feature == "point_pair_features":
                num_feat += 4
            elif feature == "relative_position" or feature == "relative_velocity":
                num_feat += 2
            else:
                num_feat += 1

        if self.E_feat is None:
            self.E_feat = np.empty([num_edges, num_feat])

        # extract features
        for i, edge in enumerate(self.E):
            # iterate through all edges and extract point pair data
            X_i = self.X[int(edge[0]), :].reshape(self.X.shape[1], 1)
            X_j = self.X[int(edge[1]), :].reshape(self.X.shape[1], 1)

            V_i = self.V[int(edge[0]), :].reshape(self.V.shape[1], 1)
            V_j = self.V[int(edge[1]), :].reshape(self.V.shape[1], 1)

            feat = []
            for feature in features:
                # extract desired features of each edge

                if feature == "point_pair_features":
                    d, theta_v1_v2, theta_d_v1, theta_d_v2 = get_En_equivariant_point_pair_metrics(
                        X_i, X_j, V_i, V_j, edge_mode)
                    feat.extend([d, theta_v1_v2, theta_d_v1, theta_d_v2])

                elif feature == "spatial_euclidean_distance":
                    d = np.linalg.norm(X_i - X_j, ord=2)
                    feat.append(d)

                elif feature == "velocity_euclidean_distance":
                    dv = np.linalg.norm(V_i - V_j, ord=2)
                    feat.append(dv)

                elif feature == "relative_position":
                    # relative position from source node w.r.t. to target node of the edge
                    # because in GNN layer this edge is used for target node update (all incoming edges)
                    if edge_mode == "directed":
                        dx = X_i[0, 0] - X_j[0, 0]
                        dy = X_i[1, 0] - X_j[1, 0]
                    elif edge_mode == "undirected":
                        dx = abs(X_i[0, 0] - X_j[0, 0])
                        dy = abs(X_i[1, 0] - X_j[1, 0])

                    feat.extend([dx, dy])

                elif feature == "relative_velocity":
                    if edge_mode == "directed":
                        du = V_i[0, 0] - V_j[0, 0]
                        dv = V_i[1, 0] - V_j[1, 0]
                    elif edge_mode == "undirected":
                        du = abs(V_i[0, 0] - V_j[0, 0])
                        dv = abs(V_i[1, 0] - V_j[1, 0])

                    feat.extend([du, dv])

                else:
                    raise Exception("Invalid feature specified")

            # store features in edge feature matrix
            self.E_feat[i, :] = np.array(feat).reshape(1, len(feat))

    def extract_single_node_features(self, features: List[str]) -> None:
        """ Extracts node features.

        Saves the features of each node in the node feature matrix X_feat.

        Args:
            features: Specifies which features to extract
                - rcs                         SE(n) invariant
                - degree                      SE(n) invariant
                - time_index                  SE(n) invariant
                - velocity_vector_length      SE(n) invariant
                - velocity_vector             T(n) invariant
                - spatial_coordinates         NOT invariant
        """

        # get total number of features to extract
        num_feat = 0

        for feature in features:
            if feature == "velocity_vector" or feature == "spatial_coordinates":
                num_feat += 2
            elif feature == "degree":
                self.add_degree_to_inv_features()
                num_feat += 1
            else:
                num_feat += 1

        for feature in features:
            if feature == "rcs":
                feat = self.F.get("rcs")

            elif feature == "time_index":
                feat = self.F.get("time_index")

            elif feature == "degree":
                feat = self.F.get("degree")

            elif feature == "velocity_vector_length":
                feat = np.linalg.norm(self.V, ord=2, axis=1).reshape(
                    self.V.shape[0], 1)

            elif feature == "velocity_vector":
                feat = self.V

            elif feature == "spatial_coordinates":
                feat = self.X

            if self.X_feat is None:
                self.X_feat = feat
            else:
                self.X_feat = np.concatenate((self.X_feat, feat), axis=1)

    def show(self, node_size: float = 60, show_velocity_vector: bool = False, vec_scale: float = 10, with_labels: bool = False) -> None:
        """ show geometric graph with nodes at spatial X coordinates.
        """
        G = nx.Graph()

        # define nodes
        for i, x in enumerate(self.X):
            G.add_node(i, pos=x[0:2])

        pos = nx.get_node_attributes(G, 'pos')

        # define edges
        for i, a in enumerate(self.A):
            for j, bool in enumerate(a):
                if bool == 1:
                    G.add_edge(i, j)

        # plot geometric graph
        _, ax = plt.subplots()
        nx.draw(G, pos, ax=ax, node_size=node_size, with_labels=with_labels)
        plt.axis('on')  # turns on axis
        ax.tick_params(left=True, bottom=True,
                       labelleft=True, labelbottom=True)
        if show_velocity_vector:
            ax.quiver(self.X[:, 0], self.X[:, 1], self.V[:, 0],
                      self.V[:, 1], scale=vec_scale, color="red")
