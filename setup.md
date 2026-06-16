# 1. See what projects exist
aws s3 ls --no-sign-request s3://fcp-indi/data/Projects/

# 2. Look inside the one you want (CoRR example)
aws s3 ls --no-sign-request s3://fcp-indi/data/Projects/CORR/

# 3. Download it
aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/CORR/ ./corr_data/

setup github repository connection 


import networkx as nx
G = nx.read_graphml("yourfile.graphml")
A = nx.to_numpy_array(G, weight="weight")  # adjacency/connectivity matrix

