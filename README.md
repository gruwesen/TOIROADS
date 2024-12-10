# TOIROADS
This is a tool for generating timeless traffic-like graphs suitable for congestion prediction.

Make graph dataset with tools found in
RoadGraphGeneratorTOI.py

Make a loader using
loader.py

Make manipulations on an existing graph with tools found in 
graphmanipulator.py

A minimal tutorial is found in 
Notebook_for_TØIRoads.ipynb

## The Algorithm
An algorithm for the main graph generation is found in the paper TØIRoads: A Road Data Model Generation Tool by Grunde Wesenberg and Ana Ozaki.
This corresponds to lines 138-161 in RoadGraphGeneratorTOI.py, GraphMaker.generate_graphs().
TorchDatasetMaker.generate_graphs_with_congestion collects the output network into a torch dataset.

## Making datasets
Import RoadGraphGeneratorTOI as RGG
RGG.TorchDataSetMaker.generate_torch_dataset_with_congestion makes a list of pytorch geometric Data objects, each containing a graph dataset as per the graph generation specifications.
It uses RGG.GraphMaker.generate_graphs to generate each single graph.

## Associated Article
This code is associated with the paper "TØIRoads: A Road Data Model Generation Tool", accepted for publication in TGDK, Volume 2, Issue 2 (2024 or 2025).
DOI or publication details will be added when available.