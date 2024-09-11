
# Commented out IPython magic to ensure Python compatibility.
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# %matplotlib inline

G = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)  # method to create a random undirected graph of 10 units, with probability of 0.3 of having edges between units/ nodes

plt.figure()
plt.axis('off')
nx.draw_networkx(G,
                 pos=nx.spring_layout(G, seed=0),
                 node_size=600,
                 cmap='coolwarm',
                 font_size=14,
                 font_color='white')

def random_walk(start, length):
  walk = [str(start)]   # starting node

  for i in range(length):
    neighbors = [node for node in graph.neighbors(start)]
    next_node = np.random.choice(neighbors, 1)[0]
    walk.append(str(next_node))
    start = next_node

  return walk

# create a list of random walks
#print(random_walk(0, 10))

graph = nx.karate_club_graph()

graph.nodes

graph.nodes[2], graph.nodes[9]

# processing labels (Mr. Hi = 0, officer = 1)
labels = []
for node in graph.nodes:
  label = graph.nodes[node]['club']
  labels.append(1 if label== 'officer' else 0)

# plotting the graph

plt.figure(figsize=(12, 22))
plt.axis('off')
nx.draw_networkx(graph,
                 pos=nx.spring_layout(graph, seed=0),
                 node_size=600,
                 cmap='coolwarm',
                 font_size=14,
                 font_color='white')

# creating a dataset of a list of random walks
walks = []
for node in graph.nodes:
  # 80 random walks of length 10 for every node
  for _ in range(80):
    walks.append(random_walk(node, 10))

# print the first random walk
print(walks[0])

walks

# create word2vec
model = Word2Vec(walks,
                 hs=1,    # hierarchical softmax
                 sg=1,    # skipgram
                 vector_size=100,
                 window=10,
                 workers=1,
                 seed=1)

print(f'shape of embedding matrix: {model.wv.vectors.shape}')

# build vocabulary
model.build_vocab(walks)

# train model
model.train(walks, total_examples=model.corpus_count, epochs=30, report_delay=1)

# finding the most similar nodes (using cosine similarity)
print('nodes most similar to node 0: ')
for similarity in model.wv.most_similar(positive=['0']):
  print(f'{similarity}')

# checking the similarity between two nodes
print(f"\nSimilarity between nodes 0 and 4: {model.wv.similarity('0', '4')}")
print(f"\nSimilarity between nodes 0 and 1: {model.wv.similarity('0', '1')}")

# node embedding of node 2
model.wv.get_vector('2')

# training a TSNE model to classify the nodes into 2 categories
from sklearn.manifold import TSNE

# preprocess word vectors and labels
nodes_wv = np.array([model.wv.get_vector(str(i)) for i in range(len(model.wv))])
labels = np.array(labels)

# train TSNE
tsne = TSNE(n_components=2,
            learning_rate='auto',
            init='pca',
            random_state=0).fit_transform(nodes_wv)

# plotting the TSNE
plt.figure(figsize=(6, 6))
plt.scatter(tsne[:, 0], tsne[:, 1], s=100, c=labels, cmap='coolwarm')
plt.show()

# creating a random forest classifier to check the performance of our model

# separating the dataset
train_mask = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
test_mask = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33]

nodes_wv[train_mask].shape

# train classifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(nodes_wv[train_mask], labels[train_mask])

# evaluate accuracy
y_pred = classifier.predict(nodes_wv[test_mask])
acc = accuracy_score(y_pred, labels[test_mask])
print(f'accuracy: {acc * 100:.2f}%')





