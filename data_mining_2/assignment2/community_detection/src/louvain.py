# please use Louvain algorithm to finish the community detection task
# Do not change the code outside the TODO part
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
# you can try different random seeds to get the best result

import networkx as nx
import csv
# you can use basic operations in networkx
# you can also import other libraries if you need, but do not use any community detection APIs

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
def store_result(G):
    with open('../data/predictions_louvain.csv', 'w') as output:
        output.write("id,category\n")
        for i in range(NUM_NODES):
            output.write("{},{}\n".format(i, G._node[i]['category']))

def initial_community(G):
    print('Initialize Community')
    com = {}
    for node in G.nodes:
        category = G.nodes[node]['category']
        com[category] = {}
        com[category]['in'] = 0
        in_weights = out_weights = 0
        for neighbor in G.neighbors(node):
            out_weights += G[node][neighbor]['weight']
            if neighbor == node:
                com[category]['in'] = G[node][neighbor]['weight']
        for predecessor in G.predecessors(node):
            in_weights += G[predecessor][node]['weight']
        com[category]['tot_in'] = in_weights
        com[category]['tot_out'] = out_weights
    return com

def calculate_delta_q(G, community, node_i, node_j, m):
    in_weights = sum([G[pred][node_i]['weight'] for pred in G.predecessors(node_i)])
    out_weights = sum([G[node_i][neigh]['weight'] for neigh in G.neighbors(node_i)])
    k_in_i = sum([G[node_i][neigh]['weight'] for neigh in G.neighbors(node_i) if G.nodes[node_j]['category'] == G.nodes[neigh]['category'] and node_i != neigh]) + sum([G[pred][node_i]['weight'] for pred in G.predecessors(node_i) if G.nodes[node_j]['category'] == G.nodes[pred]['category']])
    k_in_j = sum([G[node_i][neigh]['weight'] for neigh in G.neighbors(node_i) if G.nodes[node_i]['category'] == G.nodes[neigh]['category'] and node_i != neigh]) + sum([G[pred][node_i]['weight'] for pred in G.predecessors(node_i) if G.nodes[node_i]['category'] == G.nodes[pred]['category']])
    
    delta_q = (k_in_i / m) - (
                (community[G.nodes[node_j]['category']]['tot_in'] * out_weights + community[G.nodes[node_j]['category']][
                    'tot_out'] * in_weights) / (m ** 2))
    delta_q -= (community[G.nodes[node_i]['category']]['tot_in'] * out_weights + community[G.nodes[node_i]['category']][
        'tot_out'] * in_weights) / (m ** 2) - k_in_j / m
    data_list = [delta_q, k_in_i, k_in_j, in_weights, out_weights]
    return data_list

def merge_community(G, community):
    print('Merge community')
    m = sum([w for (u, v, w) in G.edges.data('weight')])
    for node in G.nodes:
        best_delta_q, better_community = 0, G.nodes[node]['category']
        for neighbor in G.neighbors(node):
            if G.nodes[node]['category'] != G.nodes[neighbor]['category']:
                delta_q, in_weights, out_weights, k_in_i, k_in_j = calculate_delta_q(G, community, node, neighbor, m)
                if delta_q > best_delta_q:
                    best_delta_q, best_in_weights, best_out_weights, best_k_in_i, best_k_in_j, better_community = delta_q, in_weights, out_weights, k_in_i, k_in_j, G.nodes[neighbor]['category']
        for predecessor in G.predecessors(node):
            if G.nodes[node]['category'] != G.nodes[predecessor]['category']:
                delta_q, in_weights, out_weights, k_in_i, k_in_j = calculate_delta_q(G, community, node, predecessor, m)
                if delta_q > best_delta_q:
                    best_delta_q, best_in_weights, best_out_weights, best_k_in_i, best_k_in_j, better_community = delta_q, in_weights, out_weights, k_in_i, k_in_j, G.nodes[predecessor]['category']
        # move node to better community
        if better_community != G.nodes[node]['category']:
            curr_category = G.nodes[node]['category']
            community[curr_category]['tot_in'] -= best_in_weights
            community[curr_category]['tot_out'] -= best_out_weights
            community[curr_category]['in'] -= best_k_in_i
            community[better_community]['tot_in'] += best_in_weights
            community[better_community]['tot_out'] += best_out_weights
            community[better_community]['in'] += best_k_in_j
            G.nodes[node]['category'] = better_community

def generate_supergragh(G,category):
    print('generate supergraph')
    supergragh = nx.DiGraph()
    for i ,(node,cat) in enumerate(category.items()):
        category[node] = G.nodes[cat]['category']
        supergragh.add_node(category[node],category=category[node])
    for u in G.nodes:
        cat_u = category[u]
        supergragh.add_node(cat_u,category=cat_u)
    for u,v,w in G.edges.data('weight'):
        category_u = G.nodes[u]['category']
        category_v = G.nodes[v]['category']
        if not supergragh.has_edge(category_u,category_v):
            supergragh.add_edge(category_u,category_v,weight=0)
        supergragh[category_u][category_v]['weight'] += w
    return supergragh

def louvain(G):
    category = {}
    for node in G.nodes:
        category[node] = node
        G.nodes[node]['category'] = node
    G.add_weighted_edges_from([(u, v, 1) for u, v in G.edges])
    community = initial_community(G)
    new_graph= G
    while True:
        merge_community(new_graph, community)
        num_old = new_graph.number_of_nodes()
        print(num_old)
        new_graph = generate_supergragh(new_graph, category)
        community = initial_community(new_graph)
        num_new = new_graph.number_of_nodes()
        if num_new == num_old or num_new < 6:
            break
    true_category = {node: n for n, node in enumerate(new_graph.nodes)}
    for node in G.nodes:
        G.nodes[node]['category'] = true_category[new_graph.nodes[category[node]]['category']] 
    return G

def main():
    G = getGraph() 
    G_louvain= louvain(G)     
    store_result(G_louvain)

if __name__ == "__main__":
    main()