# please use node2vec algorithm to finish the link prediction task
# Do not change the code outside the TODO part

import networkx as nx
import csv
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

# TODO: finish the class Node2Vec
import numpy as np
from gensim.models import Word2Vec
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
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
        kwargs = {"sentences": self.walks, "min_count": 0, "vector_size": 128, "sg": 1, "hs": 0, "workers": 3, "window": 3,
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
    
    def get_embeddings(self, w2v_model, graph):
        count = 0
        invalid_word = []
        _embeddings = {}
        for word in graph.nodes():
            if word in w2v_model.wv:
                _embeddings[word] = w2v_model.wv[word]
            else:
                invalid_word.append(word)
                count += 1
        self._embeddings = _embeddings
 
        return self._embeddings
 
    

    def train(self):
        kwargs = {"sentences": self.walks, "min_count": 0, "vector_size": 128, "sg": 1, "hs": 0, "workers": 3, "window": 3,
              "epochs": 3}
        self.model = Word2Vec(**kwargs)
        return self.model
    
    def predict(self, source, target):

        if source not in self._embeddings:
            return 0
        if target not in self._embeddings:
            return 0
        enc1 = self._embeddings[source]
        enc2 = self._embeddings[target]
        inp = torch.Tensor(np.array([enc2 - enc1]))
        # use embeddings to predict links
        prob = self._classifier(inp).detach().numpy()[0][1]

        return prob
    
    def train_classifier(self):
        # use torch to train a classifier
        # define a classifier
        class Classifier(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                # self.fc2 = nn.Linear(hidden_size, hidden_size)
                # self.fc3 = nn.Linear(hidden_size, output_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
            def forward(self, x):
                x = F.relu(self.fc1(x))
                # x = F.relu(self.fc2(x))
                # x = self.fc3(x)
                x = self.fc2(x)
                x = F.softmax(x, dim=1)
                return x
            
        # train the classifier
        # get embeddings of each node
        embeddings = self._embeddings
        is_dataset_exist = True
        if not is_dataset_exist:
            edges = list(self.graph.edges())
            nodes = list(self.graph.nodes())
            pos_samples, neg_samples = [], []
            # make samples
            for source, target in tqdm(edges):
                pos_samples.append((source, target))
                # negative sampling
                while True:
                    neg_source = np.random.choice(nodes)
                    neg_target = np.random.choice(nodes)
                    if (neg_source, target) not in edges and (neg_source, target) not in neg_samples:
                        neg_samples.append((neg_source, target))
                        break
                    if (source, neg_target) not in edges and (source, neg_target) not in neg_samples:
                        neg_samples.append((source, neg_target))
                        break
                    if (neg_source, neg_target) not in edges and (neg_source, neg_target) not in neg_samples:
                        neg_samples.append((neg_source, neg_target))
                        break
            # make labels
            pos_labels = [1 for _ in range(len(pos_samples))]
            neg_labels = [0 for _ in range(len(neg_samples))]
            labels = pos_labels + neg_labels
            samples = pos_samples + neg_samples
            # datas = [(embeddings[source] + embeddings[target])/2 for source, target in samples]
            datas = [embeddings[target] - embeddings[source] for source, target in samples]
            # shuffle
            indices = np.arange(len(datas))
            np.random.shuffle(indices)
            datas = np.array(datas)[indices]
            labels = np.array(labels)[indices]
            # split train and validation set
            train_size = int(len(datas) * 0.8)
            train_datas, train_labels = datas[:train_size], labels[:train_size]
            val_datas, val_labels = datas[train_size:], labels[train_size:]
            # save datas
            np.save('D:/temp/datamining/assignment2/link_prediction/data/train_datas.npy', train_datas)
            np.save('D:/temp/datamining/assignment2/link_prediction/data/train_labels.npy', train_labels)
            np.save('D:/temp/datamining/assignment2/link_prediction/data/val_datas.npy', val_datas)
            np.save('D:/temp/datamining/assignment2/link_prediction/data/val_labels.npy', val_labels)
        else:
            train_datas = np.load('D:/temp/datamining/assignment2/link_prediction/data/train_datas.npy')
            train_labels = np.load('D:/temp/datamining/assignment2/link_prediction/data/train_labels.npy')
            val_datas = np.load('D:/temp/datamining/assignment2/link_prediction/data/val_datas.npy')
            val_labels = np.load('D:/temp/datamining/assignment2/link_prediction/data/val_labels.npy')
            # print(sum(train_labels[train_labels == 1]), sum(train_labels[train_labels == 0]))
        # train
        is_trained = False
        if not is_trained:
            print('start training')
            classifier = Classifier(128, 64, 2)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
            epochs = 10
            for epoch in range(epochs):
                pbar = tqdm(range(len(train_datas)))
                running_loss = 0.0
                # shuffle train datas
                indices = np.arange(len(train_datas))
                np.random.shuffle(indices)
                train_datas = train_datas[indices]
                train_labels = train_labels[indices]
                for i in pbar:
                    optimizer.zero_grad()
                    output = classifier(torch.Tensor([train_datas[i]]))
                    # print(output, train_labels[i],train_datas[i][0])
                    loss = criterion(output, torch.LongTensor([train_labels[i]]))
                    loss.backward()
                    running_loss += loss.item()
                    optimizer.step()
                    pbar.set_postfix({'epoch': epoch, 'loss': loss.item(), 'running_loss': running_loss/(i+1)})
                # evaluate
                correct = 0
                total = 0
                with torch.no_grad():
                    for i in range(len(val_datas)):
                        output = classifier(torch.Tensor([val_datas[i]]))
                        predict = output.argmax().item()
                        total += 1
                        correct += (predict == val_labels[i])
                print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))
            # save model
            torch.save(classifier.state_dict(), 'D:/temp/datamining/assignment2/link_prediction/data/classifier.pt')
        else:
            print('loading trained model')
            classifier = Classifier(128, 64, 2)
            classifier.load_state_dict(torch.load('D:/temp/datamining/assignment2/link_prediction/data/classifier.pt'))
        self._classifier = classifier

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
       
        
    
        
        

### TODO ###
### you can define some useful functions here if you want


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

    model = Node2Vec(graph=G)

    model.train()

    embeddings = model.get_embeddings(model.model,model.graph)

    model.train_classifier()

    store_result(model)

if __name__ == "__main__":
    main()