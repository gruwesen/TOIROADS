from torch_geometric.data import InMemoryDataset, DataLoader
from multiprocessing import Pool, cpu_count
import RoadGraphGeneratorTOI as RGG
from graphmanipulator import GraphMutator, GraphUtils
import torch
import time
import os
import datetime
import pickle
 


class RUDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None, pre_transform=None):
        self.data_list = data_list
        super(RUDataset, self).__init__('.', transform, pre_transform)
        self.data, self.slices = self.collate(data_list)

    @property
    def processed_file_names(self):
        # Dummy override, as required by the API
        return ['dummy.pt']
    
    def process(self):
        #This can be used when loading from files.
        pass



class MyOwnDataset(InMemoryDataset):
    #This implementation is a hack, working together with the logic in make_loader to make a dataset based on RGG graph lists.
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['dummy_data.pt']
    
    def load(self):
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process_dummy(self, data_list):
        # Read data into huge `Data` list.
        data_list = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def make_loader_multiprocess(n_graphs, n_nodes, batch_size, dataset_name, shuffle=False, **extra_args):
    #Makes a dataloader with normalized graph data.
    tic = time.time()
    data_train = MyOwnDataset("./data/", None, None)

    if os.path.exists("./data/datasets/{}.data".format(dataset_name)):
        print("Dataset with the name {} exists. Loading dataset.".format(dataset_name))
        dataset = torch.load("./data/datasets/{}.data".format(dataset_name))
    else:
        print("Dataset with the name {} doesn't exist. Generating dataset.".format(dataset_name))
        dataset = generate_parallel_torch_dataset(n_graphs, **extra_args)
        torch.save(dataset, "./data/datasets/{}.data".format(dataset_name))
        with open("./data/datasets/{}.README".format(dataset_name), 'a') as f:
            f.write("Dataset {}, {} graphs of {} nodes".format(dataset_name, n_graphs, n_nodes))
            f.write("\n" + "datetime.now(): " + str(datetime.datetime.now()))

    data_train.process_dummy(dataset)
    data_train.load()
    toc = time.time()
    print("Loader generation time: ", toc-tic)

    return DataLoader(data_train, batch_size=batch_size, shuffle=shuffle)


def generate_graphs_wrapper(args):
    n_graphs, extra_args = args

    return RGG.TorchDataSetMaker.generate_torch_dataset_with_congestion(n_graphs, **extra_args)

def generate_parallel_torch_dataset(n_graphs, **extra_args):
    # Determine how many processes you want to use
    num_processes = cpu_count() 
    print("Number of processes: ", num_processes)

    # Divide the total number of graphs to generate among processes
    graphs_per_process = n_graphs // num_processes
    residual = n_graphs % num_processes
    args = [(graphs_per_process + (1 if i < residual else 0 ), extra_args) for i in range(num_processes)]


    print(args)
    # Generate graphs in parallel
    with Pool(num_processes) as p:
        all_graphs = p.map(generate_graphs_wrapper, args)

    # Flatten the list of lists
    all_graphs = [graph for sublist in all_graphs for graph in sublist]
    return all_graphs


def get_graph_to_be_drawn(filename, number):
    data = torch.load(f"./data/datasets/{filename}.data")
    return data[number]


class Loader():
    def __init__(self):
        pass

    @staticmethod
    def make_variational_loaders(original_graph, n_graphs, batch_size, dataset_name, shuffle=False, **options):
        #This method returns two loaders full of variations of an original graph, plus the original graph itself.
        tic = time.time()
        
        if os.path.exists("./data/datasets/{}.pkl".format(dataset_name)):
            with open("./data/datasets/{}.pkl".format(dataset_name), 'rb') as f:
                loaded_object = pickle.load(f)
                existing = True
        else:
            existing = False

        #If the seed graph is the same, we can use existing dataset.
        if existing and os.path.exists("./data/datasets/{}.data".format(dataset_name)): 
            print("Dataset with that name {} exists. Loading dataset.".format(dataset_name))
            dataset = torch.load("./data/datasets/{}.data".format(dataset_name))
       
        elif os.path.exists("./data/datasets/{}.data".format(dataset_name)):
            print("Dataset name {} exists, but no original exists. Not loading, try using another name.".format(dataset_name))
            
        else:
            print("Dataset with the name {} doesn't exist. Generating dataset.".format(dataset_name))

            dataset = Loader.generate_parallel_torch_dataset(original_graph, n_graphs, **options)
            torch.save(dataset, "./data/datasets/{}.data".format(dataset_name))
            with open("./data/datasets/{}.README".format(dataset_name), 'a') as f:
                f.write("Dataset {}, {} graphs".format(dataset_name, n_graphs))
                f.write("\n" + "datetime.now(): " + str(datetime.datetime.now()))
            with open("./data/datasets/{}.pkl".format(dataset_name), 'wb') as f:
                pickle.dump(original_graph, f, pickle.HIGHEST_PROTOCOL)

        rudata = RUDataset(dataset)
        toc = time.time()
        print("Loader generation time: ", toc-tic)

        return DataLoader(rudata, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def load_existing_dataset(dataset_name, batch_size, shuffle=False):
        if os.path.exists("./data/datasets/{}.data".format(dataset_name)): 
            dataset = torch.load("./data/datasets/{}.data".format(dataset_name))
        else:
            print(f"Dataset {dataset_name} doesn't exist")
            
        rudata = RUDataset(dataset)
        return DataLoader(rudata, batch_size=batch_size, shuffle=shuffle)


    @staticmethod
    def generate_variational_dataset(original_graph, n_graphs, n_nodes, capacity_factor, density):
        pass
    
    @staticmethod
    def generate_graphs_wrapper(kwargs):
        
        #GraphMutator makes a generator, this needs untangling before pooling.
        return [graph for graph in GraphMutator.generate_dataset_from_graph(**kwargs)]
    
    @staticmethod
    def generate_parallel_torch_dataset(original_graph, number, **kwargs):
        # Determine how many processes you want to use
        num_processes = cpu_count() 
        print("Number of processes: ", num_processes)

        # Divide the total number of graphs to generate among processes
        graphs_per_process = number // num_processes
        residual = number % num_processes
        
        args = [{'n': graphs_per_process + (1 if i < residual else 0 ), 'original_graph': original_graph, **kwargs} for i in range(num_processes)]


        print(args)
        # Generate graphs in parallel
        with Pool(num_processes) as p:
            all_graphs = p.map(Loader.generate_graphs_wrapper, args)

        # Flatten the list of lists
        all_graphs = [graph for sublist in all_graphs for graph in sublist]
        return all_graphs