# please use Louvain algorithm to finish the community detection task
# Do not change the code outside the TODO part
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
# you can try different random seeds to get the best result
import networkx as nx
import csv
import random
# you can use basic operations in networkx
# you can also import other libraries if you need, but do not use any community detection APIs
NUM_NODES = 31136

# def a function to find adjacent nodes of a node
def findNeighbors(node, edges):
    neighbors = set()
    for i in range(len(edges)):
        if node in edges[i]:
            j = int(not (node == edges[i][0]))
            neighbors.add(edges[i][j])
    return neighbors

# def a function to caluculate modularity
def calcModularity(G, communities):
    modularity = 0
    m = len(G.edges())
    for community in communities:
        for i in community:
            for j in community:
                Aij = int(G.has_edge(i, j))
                ki = len(list(G.neighbors(i)))
                kj = len(list(G.neighbors(j)))
                modularity = modularity + (Aij - (ki*kj)/(2*m))
    return modularity/(2*m)

# def a function to merge two communities and update the networkx graph
def mergeCommunities(G, communities, i, j):
    for node in communities[j]:
        G._node[node].update({'community':i})
    communities[i] = communities[i].union(communities[j])
    del communities[j]

# def a function to calculate the change of modularity after merging two communities
def calcDeltaModularity(G, communities, i, j):
    deltaQ = 0
    m = len(G.edges())
    for k in range(len(communities)):
        if k == i or k == j:
            continue
        nik = 0
        nkj = 0
        for node in communities[i]:
            if str(node) in G._adj:
                nik += len(set(G._adj[str(node)]) & communities[k])
        for node in communities[j]:
            if str(node) in G._adj:
                nkj += len(set(G._adj[str(node)]) & communities[k])
        Aij = len(set(communities[i]) & set(communities[j]))
        nik = nik + Aij
        nkj = nkj + Aij
        ki = len(list(communities[i]))
        kj = len(list(communities[j]))
        Qij = (Aij - (nik*nkj)/(2*m))/m
        Qik = (-nik*ki)/(2*m)
        Qjk = (-nkj*kj)/(2*m)
        deltaQk = Qik + Qjk
        deltaQij = Qij - deltaQk
        deltaQ = deltaQ + deltaQij
    return deltaQ

# def a function to partition the graph into communities
def communityDetection(G):
    m = len(G.edges())
    nodes = list(G.nodes())
    communities = [[i] for i in range(len(nodes))]
    random.seed(0)
    random.shuffle(nodes)
    for node in nodes:
        maxDeltaQ = 0
        bestCommunity = -1
        neighbors = findNeighbors(node, list(G.edges()))
        currCommunity = G._node[node]['community']
        nik = 0
        for neighbor in neighbors:
            if currCommunity == G._node[neighbor]['community']:
                nik += 1
        for i in range(len(communities)):
            if i != currCommunity:
                nk = len(communities[i])
                nkj = 0
                for neighbor in neighbors:
                    if i == G._node[neighbor]['community']:
                        nkj += 1
                deltaQ = (nkj - nik*1.0*nk/(2*m))/m
                if deltaQ > maxDeltaQ:
                    maxDeltaQ = deltaQ
                    bestCommunity = i
        if maxDeltaQ > 0:
            mergeCommunities(G, communities, currCommunity, bestCommunity)
            G._node[node].update({'community':bestCommunity})
    return communities

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

def store_result(G):
    with open('../data/predictions_louvain.csv', 'w') as output:
        output.write("id,category\n")
        for i in range(NUM_NODES):
            output.write("{},{}\n".format(i, G._node[i]['community']))

def main():
    G = getGraph()

    # set random seed for partition of graph
    random.seed(0)

    # initialize community id for each node
    for node in G.nodes():
        G._node[node].update({'community':node})

    counter = 0
    prevModularity = -1
    while True:
        communities = communityDetection(G)
        currModularity = calcModularity(G, communities)
        print("Iteration: {}, Modularity: {}".format(counter, currModularity))
        counter += 1
        if currModularity == prevModularity:
            break
        prevModularity = currModularity

    store_result(G)

if __name__ == "__main__":
    main()
