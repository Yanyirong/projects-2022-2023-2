# please use PPR algorithm to finish the community detection task
# Do not change the code outside the TODO part
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
# you can try different random seeds to get the best result

import networkx as nx
import csv
# you can use basic operations in networkx
# you can also import other libraries if you need, but do not use any community detection APIs
import random
import matplotlib.pyplot as plt

NUM_NODES = 31136

# read edges.csv and construct the graph
def getGraph():
    G = nx.DiGraph()
    for i in range(NUM_NODES):
        G.add_node(i)
    with open("../data/lab1_edges.csv", 'r') as csvFile:
        csvFile.readline()
        csv_reader = csv.reader(csvFile)
        for row in csv_reader:
            source = int(row[0])
            target = int(row[1])
            G.add_edge(source, target)
    print("graph ready")
    return G

# save the predictions to csv file
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
def storeResult(G):
    with open('../data/predictions_PPR.csv', 'w') as output:
        output.write("id,category\n")
        for i in range(NUM_NODES):
            output.write("{},{}\n".format(i, G._node[i]['category']))

def push(G, u, r, q,beta):
    r_1, q_1 = r[:], q[:]
    r_1[u] += (1 - beta) * q[u]
    q_1[u] = beta * q[u] / 2
    for node in G.neighbors(u):
        q_1[node] += 0.5 * beta * q[u] / G.degree(u)
    return r_1, q_1


def approximate_page_rank(G):
    """Calculate the approximate PageRank of all nodes in a given graph."""
    num_nodes = G.number_of_nodes()
    threshold = 1e-5
    beta = 0.8
    s = random.choice(range(num_nodes))
    r = [0] * num_nodes
    q = [0] * num_nodes
    q[s] = 1
    q_u_div_d_u = [0] * num_nodes
    q_u_div_d_u[s] = 1 / G.degree(s)
    while max(q_u_div_d_u) >= threshold:
        u_options = [i for i in range(num_nodes) if q_u_div_d_u[i] >= threshold]
        u = random.choice(u_options)
        r, q = push(G ,u, r, q, beta)
        q_u_div_d_u = [q[i] / G.degree(i) for i in range(num_nodes)]
    ppr = [{'id': i, 'score': r[i]} for i in range(num_nodes)]
    return ppr

def sweep_cut(G, ppr_scores):
    """Calculate the sweep cut values at each node's ordering in descending order of PPR scores."""
    ppr_scores = sorted(ppr_scores, key=lambda x: x['score'], reverse=True)
    num_nodes = G.number_of_nodes()
    phis = []
    cur_cut = 0
    cur_vol = 0
    A_i = []
    in_degrees = {}
    current_category = 0
    last_phi = 0x3f3f3f3f
    for i in range(num_nodes):
        u = ppr_scores[i]['id']
        cur_vol += G.out_degree(u)
        cur_cut += G.out_degree(u)
        for node in G.neighbors(u):
            if node in A_i:
                cur_cut -= 1
            in_degrees[node] = in_degrees.get(node, 0) + 1
        cur_cut -= in_degrees.get(u, 0) 
        phi = cur_cut / cur_vol
        phis.append(phi)
        A_i.append(u)
        if phi >= last_phi:
            current_category += 1
        last_phi = phi
        G._node[u].update({'category': current_category})
    return phis



### TODO ###
### you can define some useful function here if you want
import matplotlib.pyplot as plt
import seaborn as sns

def draw(phis):
    """Plot a graph of conductance vs. node ordering using given Sweep Cut values."""
    # Set style for plot
    sns.set_style("ticks")
    plt.figure(figsize=(12, 6))
    # Set title and labels
    plt.title("Sweep Cut Analysis", fontweight="bold", fontsize=22)
    plt.xlabel("Node Ordering", fontweight="bold", fontsize=18)
    plt.ylabel("Conductance", fontweight="bold", fontsize=18)
    # Plot horizontal lines for reference
    plt.hlines(0.25, 0, len(phis), linestyles='dashed', color="black")
    plt.hlines(0.5, 0, len(phis), linestyles='dashed', color="black")
    plt.hlines(0.75, 0, len(phis), linestyles='dashed', color="black")
    # Remove borders from plot
    sns.despine()
    # Plot Sweep Cut values
    plt.plot(phis, linewidth=2)
    # Set size and save figure
    plt.tight_layout()
    plt.savefig('../data/PPR.png', dpi=300)

def PPR(G, seed):
    random.seed(seed)
    PPR_score = approximate_page_rank(G)
    phis = sweep_cut(G, PPR_score)
    draw(phis)

### end of TODO ###
def main():
    G = getGraph()
    ### TODO ###
    # implement your community detection alg. here
    PPR(G, 0x3f3f3f3f)
    ### end of TODO ###
    storeResult(G)

if __name__ == "__main__":
    main()