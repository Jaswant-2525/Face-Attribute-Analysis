import torch, torchvision, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on:", device)

root = "./data"
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

celeba = torchvision.datasets.CelebA(root=root, split='train',
                                     target_type='attr',
                                     download=True, transform=transform)

subset_size = 20000
subset = torch.utils.data.Subset(celeba, range(subset_size))
loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)

resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()
embeddings = []

with torch.no_grad():
    for imgs, _ in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        feats = resnet(imgs)
        embeddings.append(feats.cpu().numpy())

embeddings = np.vstack(embeddings)
np.save("celeba_embeddings.npy", embeddings)
print("Embeddings shape:", embeddings.shape)

pca = PCA(n_components=50)
Xp = pca.fit_transform(embeddings)
print("Reduced shape:", Xp.shape)

k = 20
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(Xp)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X2 = tsne.fit_transform(Xp[:5000])
y2 = labels[:5000]

plt.figure(figsize=(8,6))
sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=y2,
                palette='tab20', s=20, linewidth=0)
plt.title("t-SNE of CelebA Features (sampled 5k)")
plt.show()

import random
from torchvision.utils import make_grid

def show_cluster_samples(cluster_id, n=12):
    idx = np.where(labels == cluster_id)[0]
    chosen = np.random.choice(idx, size=min(n, len(idx)), replace=False)
    imgs = [celeba[i][0] for i in chosen]
    grid = make_grid(imgs, nrow=6, normalize=True)
    plt.figure(figsize=(10,4))
    plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
    plt.axis('off')
    plt.title(f"Cluster {cluster_id} — {len(idx)} images")
    plt.show()

for c in range(5):
    show_cluster_samples(c)

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

silhouette = silhouette_score(Xp, labels)
davies_bouldin = davies_bouldin_score(Xp, labels)
calinski = calinski_harabasz_score(Xp, labels)

print("Internal Evaluation Metrics ")
print(f"Silhouette Score         : {silhouette:.4f}")
print(f"Davies–Bouldin Index     : {davies_bouldin:.4f}")
print(f"Calinski–Harabasz Score  : {calinski:.2f}")

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

attrs = celeba.attr[:subset_size].numpy()
attr_names = celeba.attr_names
attr_index = attr_names.index("Smiling")
true_labels = attrs[:, attr_index]

nmi = normalized_mutual_info_score(true_labels, labels)
ari = adjusted_rand_score(true_labels, labels)

print("\nExternal Evaluation (vs 'Smiling' attribute)")
print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
print(f"Adjusted Rand Index (ARI)   : {ari:.4f}")

def cluster_purity(y_true, y_pred):
    clusters = np.unique(y_pred)
    correct = 0
    for c in clusters:
        idx = np.where(y_pred == c)[0]
        true_vals = y_true[idx]
        if len(true_vals) == 0: continue
        majority = np.bincount(true_vals).argmax()
        correct += np.sum(true_vals == majority)
    return correct / len(y_true)

y_true = (true_labels == 1).astype(int)
purity = cluster_purity(y_true, labels)

print(f"Cluster Purity (for Smiling): {purity:.4f}")