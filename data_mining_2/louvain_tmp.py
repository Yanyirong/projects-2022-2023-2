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


### TODO ###
### you can define some useful function here if you want
def initial_community(G):
    print('Generating Community')
    community = {}
    for node in G.nodes:
        category = G.nodes[node]['category']
        community[category] = {}       
        community[category]['in'] = 0 
        num_out = 0
        num_in = 0
        for neighbor in G.neighbors(node):
            num_out += G[node][neighbor]['weight']
            if neighbor == node:
                community[category]['in'] = G[node][neighbor]['weight'] 
        for predecessor in G.predecessors(node):
            num_in += G[predecessor][node]['weight']
        community[category]['tot_out'] = num_out
        community[category]['tot_in'] = num_in
    return community

# def delta_q(G,community,node1,node2,m):
#     num_in = 0
#     num_out = 0
#     n_in_a = 0
#     n_in_b = 0
#     for predecessor in G.predecessors(node1):
#         num_in += G[predecessor][node1]['weight']
#         if G.nodes[node2]['category'] == G.nodes[predecessor]['category']:
#             n_in_a += G[predecessor][node1]['weight']
#         if G.nodes[node1]['category'] == G.nodes[predecessor]['category']:
#             n_in_b += G[predecessor][node1]['weight']
#     for neighbor in G.neighbors(node1):
#         num_out += G[node1][neighbor]['weight']
#         if G.nodes[node2]['category'] == G.nodes[neighbor]['category'] and node1 != neighbor:
#             n_in_a += G[node1][neighbor]['weight']
#         if G.nodes[node1]['category'] == G.nodes[neighbor]['category'] and node1 != neighbor:
#             n_in_b += G[node1][neighbor]['weight']
#     q = n_in_a/m-(community[G.nodes[node2]['category']]['tot_in']*num_out+community[G.nodes[node2]['category']]['tot_out']*num_in)/(m*m)
#     p = (community[G.nodes[node1]['category']]['tot_in']*num_out+community[G.nodes[node1]['category']]['tot_out']*num_in)/(m*m)-n_in_b/m
#     return q-p,num_in,num_out,n_in_a,n_in_b

def delta_q(G, community, node1, node2, m):
    num_in, num_out, n_in_a, n_in_b = 0, 0, 0, 0
    for p in G.predecessors(node1):
        w = G[p][node1]['weight']
        num_in += w
        n_in_a += w if G.nodes[node2]['category'] == G.nodes[p]['category'] else 0
        n_in_b += w if G.nodes[node1]['category'] == G.nodes[p]['category'] else 0
    for n in G.neighbors(node1):
        if n == node1:
            continue
        w = G[node1][n]['weight']
        num_out += w
        n_in_a += w if G.nodes[node2]['category'] == G.nodes[n]['category'] else 0
        n_in_b += w if G.nodes[node1]['category'] == G.nodes[n]['category'] else 0
    tot_in, tot_out = community[G.nodes[node1]['category']]['tot_in'], community[G.nodes[node1]['category']]['tot_out']
    q = n_in_a / m - (tot_in * num_out + tot_out * num_in) / (m * m)
    tot_in, tot_out = community[G.nodes[node2]['category']]['tot_in'], community[G.nodes[node2]['category']]['tot_out']
    p = (tot_in * num_out + tot_out * num_in) / (m * m) - n_in_b / m
    return q - p, num_in, num_out, n_in_a, n_in_b


def allocation_community(G,community):
    print('allocate community')
    m=0
    for (u,v,w) in G.edges.data('weight'):
        m += w
    for node in G.nodes:
        max_delta_q = 0
        max_num_in = 0
        max_num_out = 0
        max_n_in_a = 0
        max_n_in_b = 0
        better_com = G.nodes[node]['category']
        for predecessor in G.predecessors(node):
            if G.nodes[node]['category']!=G.nodes[predecessor]['category']:
                q,num_in,num_out,n_in_a,n_in_b = delta_q(G,community,node,predecessor,m)
                if q>max_delta_q:
                    max_delta_q,max_num_in,max_num_out,max_n_in_a,max_n_in_b = q,num_in,num_out,n_in_a,n_in_b
                    better_com = G.nodes[predecessor]['category']
        
        for neighbor in G.neighbors(node):
            if G.nodes[node]['category']!=G.nodes[neighbor]['category']:
                q,num_in,num_out,n_in_a,n_in_b = delta_q(G,community,node,neighbor,m)
                if q>max_delta_q:
                    max_delta_q,max_num_in,max_num_out,max_n_in_a,max_n_in_b = q,num_in,num_out,n_in_a,n_in_b
                    better_com = G.nodes[neighbor]['category']
        
        if better_com != G.nodes[node]['category']:
            community[G.nodes[node]['category']]['tot_out'] -= max_num_out
            community[G.nodes[node]['category']]['tot_in'] -= max_num_in
            community[G.nodes[node]['category']]['in'] -= max_n_in_b

            community[better_com]['tot_out'] += max_num_out
            community[better_com]['tot_in'] += max_num_in
            community[better_com]['in'] += max_n_in_a
        G.nodes[node]['category']=better_com



def generate_supergragh(G,category):
    print('generate supergraph')
    supergragh = nx.DiGraph()
    for node in category.keys():
        category[node] = G.nodes[category[node]]['category']
    for node in G.nodes:
        supergragh.add_node(G.nodes[node]['category'])
    for node in supergragh.nodes:
        supergragh.nodes[node]['category'] = node
    for (u, v, w) in G.edges.data('weight'):
        supergragh.add_edge(G.nodes[u]['category'], G.nodes[v]['category'], weight=0)
    for (u, v, w) in G.edges.data('weight'):
        supergragh[G.nodes[u]['category']][G.nodes[v]['category']]['weight'] += w
    return supergragh

### end of TODO ###


def detect_community(G):
    category = {}
    for node in G.nodes:
        category[node] = node
        G.nodes[node]['category'] = node
    for edge in G.edges:
        G.edges[edge[0], edge[1]].update({'weight': 1})
    community = initial_community(G)
    G_new = G
    while True:
        allocation_community(G_new, community)
        num_old = G_new.number_of_nodes()
        print(num_old)
        G_new = generate_supergragh(G_new, category)
        community = initial_community(G_new)
        num_new = G_new.number_of_nodes()
        if num_new == num_old or num_new < 6:
            break
    sort_category = {}
    n = 0
    for node in G_new.nodes:
        sort_category[node] = n
        n += 1 
    for node in G.nodes:
        G.nodes[node]['category'] = sort_category[G_new.nodes[category[node]]['category']]
    return G

def main():
    G = getGraph() 
    G_detected = detect_community(G)     
    store_result(G_detected)

if __name__ == "__main__":
    main()