# please use node2vec algorithm to finish the link prediction task
# Do not change the code outside the TODO part

import networkx as nx
import csv
import random
import torch
import torch.utils.data as d
import torch.nn.functional as F
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np
# you can use basic operations in networkx
# you can also import other libraries if you need

# read edges.csv and construct the graph
def get_graph():
    G = nx.DiGraph()
    with open("../data/lab2_edges.csv", 'r') as csvFile:
        csvFile.readline()
        csv_reader = csv.reader(csvFile)
        for row in csv_reader:
            source = int(row[0])
            target = int(row[1])
            G.add_edge(source, target, weight=1)
    print("graph ready")
    return G

class Classifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 64)
        self.fc3 = torch.nn.Linear(64, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class Node2Vec:
    # you can change the parameters of each function and define other functions
    def __init__(self, graph, walk_length=80, num_walks=10, p=1.0, q=1.0):
        self.graph = graph
        self._embeddings = {}
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.preprocess_transition_probs()
        self.walks = self.simulate_walks(self.num_walks, self.walk_length)
        """ for i in range(len(self.walks)):
            assert self.walks[i] != None """
        kwargs = {"sentences": self.walks, "min_count": 0, "vector_size": 64, "sg": 1, "hs": 0, "workers": 3, "window": 3,
              "epochs": 3}
        self.model = Word2Vec(**kwargs)
        self._embeddings = self.get_embeddings(self.model, self.graph)
        
    def node2vec_walk(self, walk_length, start_node):
        '''Simulate a random walk starting from start node.'''
        G = self.graph
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])   
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.graph
        walks = []
        nodes = list(G.nodes())
        print ('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        return walks

    def get_alias_edge(self, src, dst):
        G = self.graph
        p = self.p
        q = self.q
        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        G = self.graph
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        alias_edges = {}
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return
    
    def train(self, embed_size):
        self.embed_size = embed_size
        kwargs = {"sentences": self.walks, "min_count": 0, "vector_size": 64, "sg": 1, "hs": 0, "workers": 3, "window": 5,
              "epochs": 3}
        self.model = Word2Vec(**kwargs)
        return self.model
    
    def get_embeddings(self, model, graph):
        for node in self.graph.nodes():
            self._embeddings[node] = self.model.wv[node]
            if node == 7238:
                print('00')
        return self._embeddings

    def train_classifier(self):
        self._classifier = Classifier(self.embed_size,64)
        loss_function = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self._classifier.parameters(),lr=0.001)
        source, destination, y = [],[],[]
        node_list = list(self.graph.nodes)
        for edge in self.graph.edges:
            source.append(list(self._embeddings[edge[0]]))
            destination.append(list(self._embeddings[edge[1]]))
            y.append(1.0)
            rubbish_edge = random.sample(node_list,2)
            if not self.graph.has_edge(rubbish_edge[0],rubbish_edge[1]):
                source.append(list(self._embeddings[rubbish_edge[0]]))
                destination.append(list(self._embeddings[rubbish_edge[1]]))
                y.append(0.0)
        x = (torch.tensor(source,dtype=torch.float)*torch.tensor(destination,dtype=torch.float))
        y = torch.tensor(y,dtype=torch.long)
        dataset = d.TensorDataset(x,y)
        dataloader = d.DataLoader(dataset=dataset,batch_size=16,shuffle=True)
        for epoch in range(10):
            print(epoch)
            for step, (batch_x, batch_y) in enumerate(dataloader):
                prediction = self._classifier(batch_x)
                loss = loss_function(prediction,batch_y)
                opt.zero_grad()
                loss.backward()       
                opt.step()
    
    def predict(self, source, target):
        if self.graph.has_node(source) and self.graph.has_node(target):
            enc_i = self._embeddings[source]
            enc_j = self._embeddings[target]
            # use embeddings to predict links
            input = (torch.tensor(list(enc_i)).reshape(1,-1)*torch.tensor(list(enc_j))).reshape(1,-1)
            prob = F.softmax(self._classifier(input),dim=1)[0][0]
            # print(self._classifier(input))
            prob = float(prob)
            return prob
        else:
            return 0.0

### TODO ###
### you can define some useful functions here if you want
def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []
    for(kk, prob) in enumerate(probs):
        q[kk] = K*prob
        if(q[kk] < 1.0):
            smaller.append(kk)
        else:
            larger.append(kk)
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q

def alias_draw(J, q):
    K = len(J)
    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
### end of TODO ###

def store_result(model):
    with open('../data/predictions.csv', 'w') as output:
        output.write("id,probability\n")
        with open("../data/lab2_test.csv", 'r') as csvFile:
            csvFile.readline()
            csv_reader = csv.reader(csvFile)
            for row in csv_reader:
                id = int(row[0])
                source = int(row[1])
                target = int(row[2])
                prob = model.predict(source, target)
                output.write("{},{:.4f}\n".format(id, prob))

def main():
    G = get_graph()
    model = Node2Vec(G, walk_length = 5,num_walks = 10)
    model.train(embed_size=64)
    embeddings = model.get_embeddings(model, G)
    model.train_classifier()
    store_result(model)

if __name__ == "__main__":
    main()