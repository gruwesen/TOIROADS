"""
Copyright 2024 Grunde Haraldsson Wesenberg

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from random import randint, choice, choices
import numpy as np
import math
import random
import copy

import torch
from torch_geometric.data import Data


class TorchDataSetMaker():
    """
    A class to generate datasets for traffic congestion analysis.

    This class contains methods to create datasets compatible with PyTorch Geometric
    for modeling traffic flow and congestion in a network graph. Each node in the graph
    has features, and the graph has edge connectivity and labels indicating congestion levels.
    """
    def __init__(self):
        """
        Initializes the TorchDataSetMaker class.
        """
        pass

    @staticmethod
    def generate_torch_dataset_with_congestion(number, min_size, max_size, capacity_factor=1, density=0.04, min_length=1, max_length=10, rate_pop=0.4, max_pop=3, rate_poi=0.3, max_poi=3, min_cap=10, max_cap=100):
        """
        Generates a dataset with congestion data for traffic modeling.

        Parameters
        ----------
        number : int
            The number of graphs to generate.
        min_size : int
            The minimum number of nodes in each graph.
        max_size : int
            The maximum number of nodes in each graph.
        capacity_factor : float, optional
            A scaling factor for the road capacities. Default is 1.
        density : float, optional
            The density of edges in the graph. Default is 0.04.
        min_length : float, optional
            The minimum road length for nodes. Default is 1.
        max_length : float, optional
            The maximum road length for nodes. Default is 10.
        rate_pop : float, optional
            The rate of nodes having non-zero population. Default is 0.4.
        max_pop : int, optional
            The maximum population value per node. Default is 3.
        rate_poi : float, optional
            The rate of nodes having non-zero population. Default is 0.3.
        max_poi : int, optional
            The maximum number of points of interest per node. Default is 3.
        min_cap : int, optional
            The minimum capacity of roads. Default is 10.
        max_cap : int, optional
            The maximum capacity of roads. Default is 100.

        Returns
        -------
        dataset : list of Data
            A list of PyTorch Geometric Data objects, where each Data object represents
            a graph. Each graph contains:
                - x : torch.Tensor
                    Node feature tensor of shape (num_nodes, 4), containing:
                    road_length, population, points of interest, capacity.
                - edge_index : torch.Tensor
                    Edge index tensor of shape (2, num_edges), representing graph connectivity.
                - y : torch.Tensor
                    Node label tensor of shape (num_nodes, 3), containing:
                    road_usage, congestion, congested (binary).

        Notes
        -----
        The function uses `GraphMaker.generate_graphs` to generate synthetic graphs. Ensure that
        `GraphMaker` is implemented and accessible in the same context.

        Examples
        --------
        >>> maker = TorchDataSetMaker()
        >>> dataset = maker.generate_torch_dataset_with_congestion(
        ...     number=10,
        ...     min_size=5,
        ...     max_size=15,
        ...     capacity_factor=1.2
        ... )
        >>> print(len(dataset))
        10
        >>> print(dataset[0].x.shape)
        torch.Size([num_nodes, 4])
        >>> print(dataset[0].edge_index.shape)
        torch.Size([2, num_edges])
        >>> print(dataset[0].y.shape)
        torch.Size([num_nodes, 3])
        """
        dataset = []
        for network in GraphMaker.generate_graphs(number, min_size, max_size, capacity_factor, density, min_length, max_length, rate_pop, max_pop, rate_poi, max_poi, min_cap, max_cap):
            node_feature_tensor = torch.Tensor([node.features for node in network])
            edge_indices = [[x.index, y] for x in network for y in x.neighbours]
            edge_index_tensor = torch.Tensor([list(list(zip(*edge_indices))[0]),list(list(zip(*edge_indices))[1])]).long()
                       
            node_label_tensor = torch.Tensor([[node.road_usage, node.congestion, node.congested] for node in network])
            
            dataset.append(Data(x = node_feature_tensor, edge_index = edge_index_tensor, y = node_label_tensor))
        return dataset
    

class GraphMaker():
    """
    A class for generating road graphs for traffic modeling.

    This class provides a method to create graphs simulating road networks.
    Each generated graph includes node features, a strongly connected adjacency
    matrix, and calculated traffic metrics such as road usage and congestion.

    Methods
    -------
    generate_graphs(number, min_size, max_size, capacity_factor=1, density=0.04, 
                    min_length=1, max_length=10, rate_pop=0.4, max_pop=3, 
                    rate_poi=0.3, max_poi=3, min_cap=10, max_cap=100):
        Generates multiple road graphs with specified properties and traffic data.
    """
    def __init__(self):
        pass

    @staticmethod
    def generate_graphs(number, min_size, max_size, capacity_factor=1, density=0.04, min_length=1, max_length=10, rate_pop=0.4, max_pop=3, rate_poi=0.3, max_poi=3, min_cap=10, max_cap=100):
        """
        Generates road graphs with traffic and congestion data.

        Parameters
        ----------
        number : int
            The number of graphs to generate.
        min_size : int
            The minimum number of nodes in each graph.
        max_size : int
            The maximum number of nodes in each graph.
        capacity_factor : float, optional
            A scaling factor for node capacities. Default is 1.
        density : float, optional
            The density of the adjacency matrix (probability of edge creation). Default is 0.04.
        min_length : int, optional
            The minimum road length for nodes. Default is 1.
        max_length : int, optional
            The maximum road length for nodes. Default is 10.
        rate_pop : float, optional
            The rate of population per node. Default is 0.4.
        max_pop : int, optional
            The maximum population value per node. Default is 3.
        rate_poi : float, optional
            The rate of points of interest per node. Default is 0.3.
        max_poi : int, optional
            The maximum number of points of interest per node. Default is 3.
        min_cap : int, optional
            The minimum capacity of nodes. Default is 10.
        max_cap : int, optional
            The maximum capacity of nodes. Default is 100.

        Yields
        ------
        network : list
            A list of nodes representing the graph with updated features and connectivity.

        Notes
        -----
        - This method corresponds to the main algorithm described in the TØIROADS paper.
        - Each graph includes the following calculated traffic data:
            - Road usage: Total demand passing through each node based on shortest paths.
            - Congestion: Ratio of road usage to capacity for each node.
        - The caller of this method is responsible for collecting and storing the yielded networks.

        Example
        -------
        >>> gm = GraphMaker()
        >>> graphs = list(gm.generate_graphs(
        ...     number=10, 
        ...     min_size=5, 
        ...     max_size=15, 
        ...     capacity_factor=1.2
        ... ))
        >>> print(len(graphs))
        10
        >>> graphs[0]  # Access the first generated graph
        """
        #Iterate over number of graphs in dataset
        for i in range(number):

            #Instantiate RoadMaker
            r = RoadMaker()
            
            #Instantiate nodes and set features. This corresponds to lines 4 and 5 in the algorithm.
            r.make_network(min_size, max_size, min_length, max_length, rate_pop, max_pop, rate_poi, max_poi, min_cap, max_cap, capacity_factor)

            #Generate adjacency matrix (strongly connected directed graph). This corresponds to lines 6-9 in the algorithm.
            r.set_neighbours(density)

            #Calculate traffic data: road usage and congestion. This corresponds to line 10 and 11 in the algorithm.
            RoadUsageGeneral(r.network)
            CongestionCalculator.set_congestion(r.network)

            #Yield the constructed graph 
            yield r.network


class Node():
    def __init__(self, index, x=1, y=1, z=1, c=1, a=0):
        self.index = index
        self.features = [x,y,z,c]
        self.a = a

class RoadMaker():
    def __init__(self):
        pass
    
    def make_network(self, min_size=4, max_size=10, min_length=1, max_length=10, rate_pop=0.4, max_pop=3, rate_poi=0.3, max_poi=3, min_cap=10, max_cap=100, capacity_factor=1):
        #This is the method that is used to make the nodes and features of a road network.
        self.network_size = randint(min_size,max_size)
        self.network = []
        self.totpop = 0
        self.totpoi = 0
        
        #Following values are made to make the nodes have the given rate of non-zero population and points of interest. 
        #Rate_pop
        try:
            zero_start_pop = max_pop - math.floor(max_pop/rate_pop) + 1
            zero_start_poi = max_poi - math.floor(max_poi/rate_poi) + 1
        except:
            raise ValueError("rate_pop and rate_poi need to be between 0 and 1, and should not approach either.")

        for i in range(self.network_size):
            #min_length should really be 1 or more.
            road_length = randint(min_length,max_length)
            population = max(0,randint(zero_start_pop,max_pop + 1))
            poi = max(0,randint(zero_start_poi,max_poi + 1))
            capacity = math.floor(randint(min_cap, max_cap)*capacity_factor)
            self.network.append(Node(i, road_length, population, poi, capacity))
            self.totpop += population
            self.totpoi += poi
        #Ensure that there's at least one node of each kind.
        if self.totpop == 0:
            self.network[randint(0,self.network_size-1)].features[1] = 1
        if self.totpoi == 0:
            self.network[randint(0,self.network_size-1)].features[2] = 1
        
    def make_dense_adjacency(self,N):
        #Create a symmetric, 0 diagonal, binary matrix to use as an adjecency matrix. 
        #Number of connections is not controlled for, so the matrix is dense.
        a = np.random.randint(0,2,size=(N,N))
        m = np.tril(a,-1) + np.tril(a, -1).T

        invalid_nodes = []
        for i in range(0,N):
            if m[i].sum() == 0:
                invalid_nodes.append(i)
        for row_index in invalid_nodes:
            change_index = randint(0,N-1)
            if change_index == row_index:
                change_index = (change_index + 1)%N
            m[row_index, change_index] = 1
            m[change_index, row_index] = 1
    
        return m

    def make_sane_adjacency(self,N):
        #Legacy code
        #Create a symmetric, 0 diagonal, binary matrix to use as an adjecency matrix. 
        #Number of connections should be 5 or fewer, so the matrix is dense.
        a = np.random.randint(0,2,size=(N,N))
        m = np.tril(a,-1) + np.tril(a, -1).T

        too_sparse = []
        too_dense = []
        for i in range(0,N):
            if m[i].sum() == 0:
                too_sparse.append(i)
            if m[i].sum() >5:
                too_dense.append(i)
        for row_index in too_sparse:
            change_index = randint(0,N-1)
            if change_index == row_index:
                change_index = (change_index + 1)%N
            m[row_index, change_index] = 1
            m[change_index, row_index] = 1
    
        for row_index in too_dense:
            ones = np.where(m[row_index] == 1)[0]
            change_ones = choices(ones, k=(len(ones)-5))
            for i in change_ones:
                m[row_index, i] = 0
                m[i, row_index] = 0

        #Check for graph connection. If not connected, add an edge between a connected and a disconnected, and repeat.
        #This part is probably causing trouble
        visited = np.zeros(N)
        counter = 0
        while self.visit_nodes(randint(0,N-1), m, visited):
            disconnected = np.where(visited == 0)[0]
            connected = np.where(visited == 1)[0]

            i = choice(disconnected)
            j = choice(connected)
            m[i,j] = 1
            m[j,i] = 1
            
            
            counter += 1
            if counter >=20:
                print(m, visited)
                return m
            
            visited = np.zeros(N)

        return m

    def make_skewed_adjacency_ground_up(self, N):
        #Adjacency matrix
        A = np.zeros((N, N), dtype=int)
        #unweighted likelihood distribution
        L = np.zeros(N)
        L[0] = 1
        #Probability distribution
        P = np.zeros(N)
        P[0] = 1
        #Edge counter
        E = np.zeros(N)

        for node in range(1, N):
            a = np.random.choice(N, p=P)
            b = node
            A[a,b] = 1
            A[b,a] = 1
            E[a] += 1
            E[b] += 1
            L[a] = 1/E[a]
            L[b] = 1/E[b]
            P = L/sum(L)

        return A, E
    
    def make_skewed_adjacency_without_main_lane(self, N):
        #This is making the same kind of strongly connected graph as make_skewed_adjacency_ground_up, but without a main bidirectional lane.
        #Adjacency matrix
        A = np.zeros((N, N), dtype=int)
        #unweighted likelihood distribution for edges to existing subgraph
        T = np.zeros(N)
        T[0] = 1
        #unweighted likelihood distribution for edges from existing subgraph
        F = np.zeros(N)
        F[0] = 1
        #Probability distribution to
        TP = np.zeros(N)
        TP[0] = 1
        #Probability distribution from
        FP = np.zeros(N)
        FP[0] = 1
        #Edge counters
        TE = np.zeros(N)
        FE = np.zeros(N)

        #Populate graph, starting with a subgraph consisting only of node 0.
        for node in range(1,N):
            #node to be added to connected subgraph
            a = node
            #Drawing two nodes from the connected subgraph
            to = np.random.choice(N, p=TP)
            fro = np.random.choice(N, p=FP)
            
            #updating adjacency matrix
            A[a,to] = 1
            A[fro,a] = 1
            
            #updating edge counters
            TE[a] += 1
            TE[to] += 1
            FE[a] += 1
            FE[fro] += 1

            #calculating new likelihoods
            T[a] = 1/TE[a]
            T[to] = 1/TE[to]
            F[a] = 1/FE[a]
            F[fro] = 1/FE[fro]

            #updating probability distributions
            TP = T/sum(T)
            FP = F/sum(F)
        
        return A, TE, FE

    def populate_skewed_adjacency_matrix(self, A, density=0.2, with_variation=True):
        # Set adjacency matrix constants. A is an N * N matrix
        N = len(A)
        maximum_edges = N*N - N
        
        # Exclude diagonal elements (self-loops)
        mask_no_self_loops = ~np.eye(N, dtype=bool)
        
        # Available positions where edges can be added (A == 0 and not on the diagonal)
        available_positions = np.where((A == 0) & mask_no_self_loops)
        total_possible_edges = available_positions[0].size
        
        # Adjust density depending on with_variation flag
        if ((with_variation == True) & (density<0.9)):
            variation_factor = random.uniform(0.9,1.1)
            density = density * variation_factor
        
        #Calculate edges to add depending on final density * maximum edges, minus the existing edges.
        number_of_edges = math.floor(density * maximum_edges)
        edges_to_add = number_of_edges - A.sum()

        if edges_to_add <= 0:
            print("Density is lower than minimum for a strongly connected adjacency matrix of this size. Returning A as is.")
            return A

        assert edges_to_add <= total_possible_edges, "Edges to add cannot exceed total possible edges"
        
        # Choose edges_to_add indices from total_possible_edges 
        indices = np.random.choice(total_possible_edges, size=edges_to_add, replace=False)
        
        # Connect indices with row and column selections in available_positions
        rows = available_positions[0][indices]
        columns = available_positions[1][indices]
        
        # Update adjacency matrix
        A[rows, columns] = 1

        assert A.sum() == number_of_edges, "The right amount of edges must be added"

        # Return adjacency matrix of size N with density as given 
        return A
    
    def populate_skewed_adjacency_matrix_old(self, A, E, density=0.2, with_variation=True):
        # The total number of edges possible is NxN - N, as the diagonal is empty. The number of edges made in the skeleton graph is 2N-2. So the total possible number of edges addable is given as N^2 -3N + 2.
        #A should be a bare bones adjacency matrix
        #E should be a vector describing number of edges per node for A.
        
        N = len(E)

        max_edges = N*N - 3*N + 2
        edges_to_add = math.floor(density * max_edges)

        
        if ((with_variation == True) & (density<0.9)):
            variation_factor = random.uniform(0.9,1.1)
            edges_to_add = math.floor(edges_to_add * variation_factor)

        new_edges = np.zeros(N, dtype=np.int64)
        for i in range(edges_to_add):
            a = np.random.randint(N)
            new_edges[a] += 1

        for i in range(N):
            if new_edges[i] + E[i] >= N-1:
                new_edges[i] = N-E[i]-1
            
            A_without_diag = np.copy(A[i])
            A_without_diag[i] = 1
            open_spaces = np.where(A_without_diag==0)[0]
            edge_indices = np.random.choice(open_spaces, size = new_edges[i], replace=False)
            for edge in edge_indices:
                if A[i][edge] != 0:
                    print("Assigning existing edge")
                A[i][edge] = 1
        
        return A



    def visit_nodes(self, index, adj, visited):
            visited[index] = 1
            for neighbour in np.where(adj[index] == 1)[0]:
                if visited[neighbour] == 0:
                    self.visit_nodes(neighbour, adj, visited)
            return 0 in visited

    def set_neighbours(self, density, directed=True):
        if directed:
            #A, E= self.make_skewed_adjacency_ground_up(self.network_size)
            A, TE, FE = self.make_skewed_adjacency_without_main_lane(self.network_size)
            self.adj = self.populate_skewed_adjacency_matrix(A, density)
        else:
            #self.adj = self.make_dense_adjacency(self.network_size)
            self.adj = self.make_sane_adjacency(self.network_size)
        for node in self.network:
            node.neighbours = np.where(self.adj[node.index] == 1)[0]
    
    def sanity_check_adjacency(self, adj):
        #intersections connects three to five roads.
        total_intersections = 0
        for row in adj:
            total_intersections += row.sum()
        return total_intersections/len(adj)
    
    def make_variations(self, network, max_step):
        for i in range(max_step):
            pass
    


class CongestionCalculator():
    def __init__(self):
        pass

    @staticmethod
    def set_capacities(network, factor):
        """This method sets capacities for the nodes in the network. It sets them between square root of network size and network size, but takes a factor as well to scale up or down."""
        assert isinstance(factor, (int, float)), "capacity factor is not a number!"
        
        size = len(network)
        floor = int(size**(0.5))
        for node in network:
            node.capacity = randint(floor, size)*factor


    @staticmethod
    def set_congestion(network, debug_mode=False):
        total_congestion = 0
        for node in network:
            node.congestion = node.road_usage/node.features[3]
            if node.congestion >= 1:
                node.congested = 1
                total_congestion += 1
            else:
                node.congested = 0
        if debug_mode:
            print(total_congestion)

#rewriting RoadUsage to handle directional graphs and non-binary pops and pois.
class RoadUsageGeneral():
    """RoadUsage takes a network from RoadMaker, and calculates and sets road usage and node score."""
    def __init__(self, network):
        self.network = network
        self.pops = [x.index for x in self.network if x.features[1]>0]
        self.pois = [x.index for x in self.network if x.features[2]>0]
        self.set_routes()
        self.pathcounter = self.roads_used()
        self.score()

    def score(self):
        for node in self.network:
            node.road_usage = self.pathcounter[node.index]

    def roads_used(self):
        pathcounter = [0]*len(self.network)
        for pop_index in self.pops:
            node = self.network[pop_index] 
            routes = node.routes
            pop_value = node.features[1]
            for poi_index in self.pois:
                poi_value = self.network[poi_index].features[2]
                usage_value = poi_value*pop_value
                pathcounter = self.retrace(pathcounter, routes, poi_index, usage_value)
                pathcounter[poi_index] += usage_value
                pathcounter[pop_index] += usage_value

        for poi_index in self.pois:
            node = self.network[poi_index]
            routes = node.routes
            poi_value = node.features[2]
            for pop_index in self.pops:
                pop_value = self.network[pop_index].features[1]
                usage_value = poi_value*pop_value
                pathcounter = self.retrace(pathcounter, routes, pop_index, usage_value)
                pathcounter[poi_index] += usage_value
                pathcounter[pop_index] += usage_value

        return pathcounter

    def retrace(self, pathcounter, route, node, usage_value):
        #This recursive function walks backwards one step at a time to the source given by nan in a route vector, adding to each node passed.
        if not np.isnan(route[node]):
            n = route[node]
            if not np.isnan(route[n]):
                pathcounter[n] += usage_value
                pathcounter = self.retrace(pathcounter, route, n, usage_value)
        return pathcounter


    def set_routes(self):
        #combine pops and pois, remove duplicates
        route_indices = self.pops + list(set(self.pois)-set(self.pops))
        for source in route_indices:
            node = self.network[source]
            if node.index != source:
                print("RoadUsageGeneral.set_routes network indexing is off.")
            node.routes = self.dijkstra(self.network, source)

    def dijkstra(self, graph, source, verbose = False):
        dist = [np.inf]*len(graph)
        previous = [np.nan]*len(graph)
        dist[source] = 0
        Q = [node.index for node in graph]
        
        while len(Q)>0:
            min_dist = np.inf
            u = np.nan
            if verbose:
                print(Q)
            for q in Q:
                #print(q, dist[q], min_dist, u)
                if dist[q]<min_dist:
                    min_dist = dist[q]
                    u = q
            Q.remove(u)
            
            for v in graph[u].neighbours:
                if v not in Q:
                    continue
                alt = dist[u] + graph[u].features[0]/2 + graph[v].features[0]/2
                if alt < dist[v]:
                    dist[v] = alt
                    previous[v] = u
            if verbose:        
                print("current: ", u)
                print("distances: ", dist)
                print("previous: ", previous)
            
        return previous


class RoadUsage():
    """RoadUsage is deprecated, use RoadUsageGenera. RoadUsage takes a network from RoadMaker, and calculates and sets road usage and node score."""
    def __init__(self, network):
        self.network = network
        self.sources = [x.index for x in self.network if x.features[1]>0]
        self.sinks = [x.index for x in self.network if x.features[2]>0]
        self.routes = [self.dijkstra(self.network, source) for source in self.sources]
        self.pathcounter = self.roads_used(self.routes, self.sinks)
        self.set_score(self.pathcounter, len(self.sources), len(self.sinks))
        
    def set_score(self, pathcounter, nsources, nsinks, binary_sources=True):
        if binary_sources:
            #This forces node features into
            for node in self.network:
                node.road_usage = pathcounter[node.index]
                node.score = node.road_usage + bool(node.features[1])*nsinks/2 + bool(node.features[2])*nsources/2

        else:
            print("Road Usage for non-binary pop and poi is not fully implemented yet. This will only count the endpoints more times.")
            for node in self.network:
                node.road_usage = pathcounter[node.index]
                node.score = node.road_usage + node.features[1]*nsinks/2 + node.features[2]*nsources/2
            

    def get_score(self):
        return self.network

    @staticmethod
    def get_network_score(network):
        empty_roads = 0
        empty_traffic_roads = 0        
        for node in network:
            if sum(node.features) == node.features[0]:
                empty_roads += 1
                if node.score > 0:
                    empty_traffic_roads += 1
        if empty_traffic_roads == 0:
            return 0
        else:
            return empty_traffic_roads/empty_roads
    
    @staticmethod
    def get_network_label(network):
        fraction = RoadUsage.get_network_score(network)
        if fraction == 0:
            return 0
        elif fraction < 0.5:
            return 1
        elif fraction <1:
            return 2
        else:
            return 3
    
    def dijkstra(self, graph, source, verbose = False):
        dist = [np.inf]*len(graph)
        previous = [np.nan]*len(graph)
        dist[source] = 0 #graph[source].features[0]/2
        Q = [node.index for node in graph]
        
        while len(Q)>0:
            min_dist = np.inf
            u = np.nan
            if verbose:
                print(Q)
            for q in Q:
                #print(q, dist[q], min_dist, u)
                if dist[q]<min_dist:
                    min_dist = dist[q]
                    u = q
            Q.remove(u)
            
            for v in graph[u].neighbours:
                if v not in Q:
                    continue
                alt = dist[u] + graph[u].features[0]/2 + graph[v].features[0]/2
                if alt < dist[v]:
                    dist[v] = alt
                    previous[v] = u
            if verbose:        
                print("current: ", u)
                print("distances: ", dist)
                print("previous: ", previous)
            
        return previous
    
    def retrace(self, pathcounter, route, node):
        #This recursive function walks backwards one step at a time to the source given by nan in a route vector, adding to each node passed.
        if not np.isnan(route[node]):
            #print(pathcounter, route, node)
            n = route[node]
            if not np.isnan(route[n]):
                pathcounter[n] += 1
                pathcounter = self.retrace(pathcounter, route, n)
        return pathcounter
        
    def roads_used(self, routes, sinks):
        #Sums number of intermediate node visits for each route vector (where source is given as nan), and for each sink.
        pathcounter = [0]*len(routes[0])
        for r in routes:
            for s in sinks:
                pathcounter = self.retrace(pathcounter, r, s)
        return pathcounter

class GraphMutator():
    def __init__(self):
        pass

    @staticmethod
    def inner_feature_permuter(graph, steps, max_pop=2, max_poi=2):
        #Determining which features on which nodes are going to get changed. This method is made with large steps values in mind.
        size = len(graph)

        poipopsplit = np.random.randint(0, steps)
        pois = np.random.choice(graph, poipopsplit, replace=False)
        pops = np.random.choice(graph, (steps - poipopsplit), replace=False)
    
        for node in pops:
            #This will change things once, changing value to 0 and 0 to value.
            if node.features[1]:
                node.features[1] = 0
            else:
                node.features[1] = np.random.randint(1,max_pop+1)
            
        for node in pois:
            if node.features[2]:
                node.features[2] = 0
            else:
                node.features[2] = np.random.randint(1,max_poi+1)
            
        return graph

    @staticmethod
    def length_permuter(graph, steps, max_length=3):
        
        nodes = np.random.choice(graph, steps, replace=False)
        
        for node in nodes:
            new_length = np.random.randint(1,max_length+1)
            if new_length != node.features[0]:
                node.features[0] = new_length
            else:
                range_except_current_length = [*range(1,node.features[0])]+[*range(node.features[0]+1,max_length+1)]
                node.features[0] = np.random.choice(range_except_current_length)
        #The graph has already been changed, no real need to return it.
        return graph

    @staticmethod
    def edge_adder(graph, steps, symmetric=False):
    
        size = len(graph)
        full_node = False
        
        while(full_node or steps>0):
            full_node = False
            
            nodes = np.random.choice(graph, steps, replace=True)
            
            for node in nodes:
                if len(node.neighbours) == size-1:
                    full_node = True
                    continue
                else:
                    steps -= 1
                    edge_index = np.random.randint(0,size)
                    if edge_index in node.neighbours:
                        i = np.where(node.neighbours == edge_index)[0][0]
                        print(edge_index, i, node.neighbours, steps)
                        if i == len(node.neighbours)-1:
                            prox_range = [*range(node.neighbours[i-1]+1,node.neighbours[i])]
                        elif i == 0:
                            prox_range = [*range(node.neighbours[i]+1, node.neighbours[i+1])]
                        else:
                            prox_range = [*range(node.neighbours[i-1]+1,node.neighbours[i])] + [*range(node.neighbours[i]+1, node.neighbours[i+1])]
                        if len(prox_range) == 0:
                            prox_range = list(set(range(0,size)) - set(node.neighbours))
                        edge_index = np.random.choice(prox_range)
                        
                    neighbour_index = np.searchsorted(node.neighbours, edge_index)
                    np.insert(node.neighbours, neighbour_index, edge_index)
                    if symmetric:
                        if node.index in graph[edge_index].neighbours:
                            continue
                        else:
                            np.insert(graph[edge_index].neighbours, np.searchsorted(graph[edge_index].neighbours, node.index), node.index)

    @staticmethod
    def edge_remover(graph, steps, symmetric=False):

        def path_exist(graph, start, target):
            #This inner function is used to check if the connected graph is still connected after a single edge removal.
            visited = [start]
            next_visit = graph[start].neighbours.tolist()
            iteration = 0
            while len(next_visit) > 0:
                new_node = next_visit[-1]
                visited.append(new_node)
                if target in graph[new_node].neighbours:
                    return True
                elif len(next_visit) == 1:
                    next_visit = list(set(graph[new_node].neighbours.tolist()) - set(next_visit) - set(visited))
                else:
                    next_visit = next_visit[:-1] + list(set(graph[new_node].neighbours.tolist()) - set(next_visit) - set(visited))
            return False


        size = len(graph)
        invalid_pairs = {}
        invalid_nodes = []
        
        faultline = steps+size
        
        while(steps>0):
            
            faultline -= 1
            if faultline <= 1:
                print("Struggling to remove more edges, returning invalid pairs")
                return invalid_pairs
            
            nodes = np.random.choice(graph, steps, replace=True)
            
            for node in nodes:
                
                if node.index in invalid_nodes:
                    print(node.index, " is an invalid node: No edge can be safely removed. Looping to find an alternative.")
                    continue
                
                if not node.index in invalid_pairs.keys():
                    invalid_pairs[node.index] = []
                
                valid_choices = list(set(node.neighbours)-set(invalid_pairs[node.index]))
                if len(valid_choices) == 0:
                    invalid_nodes.append(node.index)
                    continue
                
                edge_index = np.random.choice(valid_choices)
                
                neighbours = node.neighbours
                node.neighbours = node.neighbours[node.neighbours != edge_index]
                if not path_exist(graph, node.index, edge_index):
                    
                    invalid_pairs[node.index].append(edge_index)
                    node.neighbours = neighbours
                    continue
                    
                if symmetric:
                    neighbours = graph[edge_index].neighbours
                    graph[edge_index].neighbours = graph[edge_index].neighbours[graph[edge_index].neighbours != node.index]
                    if not path_exist(graph, edge_index, node.index):
                        
                        if not edge_index in invalid_pairs.keys():
                            invalid_pairs[edge_index] = []
                        
                        invalid_pairs[node.index].append(edge_index)
                        invalid_pairs[edge_index].append(node.index)
                        graph[edge_index].neighbours = neighbours
                        continue
                steps -= 1
    
