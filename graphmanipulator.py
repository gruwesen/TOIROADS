from copy import deepcopy
import networkx as nx
import torch
from torch_geometric.data import Data
import random
import math

class GraphMutator():
    def __init__(self):
        pass

    @staticmethod
    def generate_dataset_from_graph(original_graph, **config):
        for graph in GraphMutator.generate_mutations_from_graph(original_graph, **config):
            GraphUtils.set_road_usage_s(graph)
            GraphUtils.calculate_congestion_s(graph)
            yield GraphMutator.nx_to_data_s(graph)

    @staticmethod
    def generate_mutations_from_graph(original_graph, **config):
        """
        This method is made to convert a road graph to a networkx graph. Then it makes copies and modifies them according to config.
        The config should have 'n', 'change_percent', 'max_length', 'max_pop', 'max_poi', 'max_cap'.
        """
        
        if GraphUtils.is_road_usage_graph(original_graph):
            original_graph = GraphUtils.convert_to_networkx(original_graph)
        elif GraphUtils.is_not_networkx(original_graph):
            raise TypeError 
        
        #some things that need to be in config:
        #mutation types: 
        
        if config['mild_mutation']:    
            for i in range(config['n']):
                new_graph = deepcopy(original_graph)
                GraphMutator.mild_increasing_mutation(new_graph, **config)
                yield new_graph
        else:
            for i in range(config['n']):
                new_graph = deepcopy(original_graph)
                GraphMutator.mutate(new_graph, **config)
                yield new_graph

    @staticmethod
    def mutate(graph, **config):
        to_change = math.floor(config['change_percent'] * len(graph))
        nodes_to_change = random.sample(range(len(graph)), to_change)
        
        cap_floor = math.floor(math.sqrt(len(graph)))
        if config['change_length']:
            new_lengths = [random.randint(1, config['max_length']) for _ in range(to_change)]
        if config['change_pop']:
            new_pops = [random.randint(0, config['max_pop']) for _ in range(to_change)]
        if config['change_poi']:
            new_pois = [random.randint(0, config['max_poi']) for _ in range(to_change)]
        if config['change_cap']:
            new_caps = [random.randint(cap_floor, config['max_cap']) for _ in range(to_change)]
        
        features = {}
        for n in range(to_change):
            features[nodes_to_change[n]] = {}
            if config['change_length']:
                features[nodes_to_change[n]]['length'] = new_lengths[n]
            if config['change_pop']:
                features[nodes_to_change[n]]['population'] = new_pops[n]
            if config['change_poi']:
                features[nodes_to_change[n]]['points of interest'] = new_pois[n]
            if config['change_cap']:
                features[nodes_to_change[n]]['capacity'] = new_caps[n]
        
        #removing the following as a temporary fix
        #features = {nodes_to_change[n]: {'length': new_lengths[n], 'population': new_pops[n], 'points of interest': new_pois[n], 'capacity': new_caps[n]} for n in range(to_change)}

        nx.set_node_attributes(graph, features)

    def mild_increasing_mutation(graph, **config):
        #Might need some way to make a milder kind of mutation of the graph. This function will make a single change per 
        to_change = math.floor(config['change_percent'] * len(graph))
        nodes_to_change = random.sample(range(len(graph)), to_change)
        default_value = None
        features = dict.fromkeys(nodes_to_change, default_value)
        for n in range(to_change):
            change = 0
            if config['random_change']:
                #Sets up for change of 1 random feature out of the four. 
                change = random.randint(1,4)


            if change == 1 or config['change_length']:
                features[nodes_to_change[n]] = {'length': graph.nodes[nodes_to_change[n]].get('length') + 1}
            elif change == 2 or config['change_pop']:
                features[nodes_to_change[n]] = {'population': graph.nodes[nodes_to_change[n]].get('population') + 1}
            elif change == 3 or config['change_poi']:
                features[nodes_to_change[n]] = {'points of interest': graph.nodes[nodes_to_change[n]].get('points of interest') + 1}
            elif change == 4 or config['change_cap']:
                features[nodes_to_change[n]] = {'capacity': graph.nodes[nodes_to_change[n]].get('capacity') + 1}
        
        #Set the changed features
        nx.set_node_attributes(graph, features)
    
    def mild_decreasing_mutation(graph, **config):
        #Might need some way to make a milder kind of mutation of the graph. This function will make a single change per 
        to_change = math.floor(config['change_percent'] * len(graph))
        nodes_to_change = random.sample(range(len(graph)), to_change)

        features = {}
        for n in range(to_change):
            change = random.randint(1,4)
            if change == 1:
                features[nodes_to_change[n]] = {'length': max(1, graph.nodes[nodes_to_change[n]].get('length') - 1)}
            elif change == 2:
                features[nodes_to_change[n]] = {'population': max(0,graph.nodes[nodes_to_change[n]].get('population') - 1)}
            elif change == 3:
                features[nodes_to_change[n]] = {'points of interest': max(0,graph.nodes[nodes_to_change[n]].get('points of interest') - 1)}
            elif change == 4:
                features[nodes_to_change[n]] = {'capacity': graph.nodes[nodes_to_change[n]].get('capacity') - 1}
        
        #Set the changed features
        nx.set_node_attributes(graph, features)

            


    @staticmethod
    def nx_to_data_s(graph):
        #Converting a single RoadGNN_s graph in nx format to a torch geometric Data object.
        
        node_features = torch.tensor([[graph.nodes[n]['length'], graph.nodes[n]['population'], graph.nodes[n]['points of interest'], graph.nodes[n]['capacity']] for n in graph.nodes()], dtype=torch.float)
        edge_list = torch.tensor(list(graph.edges), dtype=torch.long).t()
        node_labels = torch.Tensor([[graph.nodes[n]['road_usage'], graph.nodes[n]['congestion'], graph.nodes[n]['congested']] for n in graph.nodes()]) 

        return Data(x = node_features, edge_index = edge_list, y = node_labels)


                
    

class GraphUtils():
    def __init__(self):
        pass

    @staticmethod
    def convert_to_networkx(roadgraph):
        #Construct base graph with edges
        G = nx.from_numpy_array(roadgraph.adj, create_using=nx.DiGraph)

        #Categorize features with names
        features = {node.index: {'length': node.features[0], 'population': node.features[1], 'points of interest': node.features[2], 'capacity': node.features[3]} for node in roadgraph.network}
        
        #Add features to graph nodes
        nx.set_node_attributes(G, features)

        GraphUtils.set_edge_weight_from_node_length(G)
        
        return G
    
    @staticmethod
    def set_edge_weight_from_node_length(graph):
        #Manipulates graph edges. In the original road graph generator, all lengths are on the nodes. This moves the property to the incoming edge, which keeps the shortest paths the same.
        for u, v, d in graph.edges(data=True):
            d['weight'] = graph.nodes[v].get('length')

    @staticmethod
    def set_road_usage_s(graph):
        #This method should set road usage on a network x graph that has length on ingoing edges
        
        #Set up road usage receptacle
        road_usage = {i: 0 for i in range(len(graph))}
        
        #loop through nodes for sources
        for u in graph.nodes():
            pop = graph.nodes[u].get('population')
            
            #go to next node if there's no population
            if pop == 0:
                continue
            
            #loop through nodes for targets
            for v in graph.nodes():
                poi = graph.nodes[v].get('points of interest')
                if poi == 0:
                    continue

                shortest_paths = [path for path in nx.all_shortest_paths(graph, u, v)]
                split_factor = len(shortest_paths)
                for path in shortest_paths:
                    for n in path:
                        road_usage[n] += pop*poi/split_factor

        #finally set road usage on graph
        nx.set_node_attributes(graph, road_usage, 'road_usage')
    
    @staticmethod
    def calculate_congestion_s(graph):
        congestion = {}
        congested = {}
        for node in graph.nodes():
            congestion = graph.nodes[node].get('road_usage')/graph.nodes[node].get('capacity')
            if congestion >= 1:
                congested = 1
            else:
                congested = 0
            nx.set_node_attributes(graph, {node: {'congestion': congestion, 'congested': congested}})
            


    @staticmethod 
    def is_road_usage_graph(graph):
        if type(graph).__name__ == "RoadMaker":
            return True
        else:
            return False
        
    @staticmethod
    def is_not_networkx(graph):
        if type(graph).__name__ == "DiGraph":
            return False
        else:
            return True
