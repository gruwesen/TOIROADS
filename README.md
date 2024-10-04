# TOIROADS
This is a tool for generating timeless traffic-like graphs suitable for congestion prediction.

Make graph dataset with tools found in
RoadGraphGeneratorTOI.py

Make a loader using
loader.py

Make manipulations on an existing graph with tools found in 
graphmanipulator.py

A small tutorial is found in 
Notebook_for_TØIRoads.ipynb

## The Algorithm
An algorithm for the main graph generation is found in the paper TØIRoads: A Road Data Model Generation Tool by Grunde Wesenberg and Ana Ozaki.
This corresponds to lines 138-161 in RoadGraphGeneratorTOI.py, GraphMaker.generate_graphs().
TorchDatasetMaker.generate_graphs_with_congestion collects the output network into a torch dataset.
