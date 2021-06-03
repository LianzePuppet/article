import torch
import matplotlib
# Don't show graph
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class config:
    def __init__(self):
        self.embedding_save_path = '../data/vectors.txt'
        self.plot_only = 500
        self.tsne_save_file='../data/Glove.png'
args = config()
def read_embedding(embedding_file: str, return_tensor: bool=True):
    token_list = []
    embedding_list = []
    with open(embedding_file, encoding='utf-8') as f:
        for line in f:
            word, embedding = line.strip().split(' ',1)
            token_list.append(word)
            embedding_list.append(list(map(float, embedding.split(' '))))
    if return_tensor:
        return token_list, torch.tensor(embedding_list)
    else:
        return token_list, embedding_list

def plot_with_labels(embeddings, nodes, filename='tsne.png'):
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(nodes):
    x, y = embeddings[ i, : ]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.savefig(filename)

word_list, final_embedding = read_embedding(args.embedding_save_path)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embeddings = tsne.fit_transform(final_embedding[ :args.plot_only, : ])
labels = word_list.copy()[:args.plot_only]
print('Visualizing.')
plot_with_labels(low_dim_embeddings, labels, filename=args.tsne_save_file)
print('TSNE visualization is completed, saved in {args.tsne_save_file}.')
