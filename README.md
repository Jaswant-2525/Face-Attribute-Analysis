📌 Face Attribute-Based Image Clustering using ResNet50 & K-Means

This project performs unsupervised image clustering on the CelebA face dataset using deep feature extraction and machine learning.
A pretrained ResNet50 model is used to extract facial embeddings, followed by PCA for dimensionality reduction and K-Means for clustering.
The aim is to group visually similar faces based on attributes such as gender, age, expression, hairstyle, presence of glasses, etc.

🔍 Project Overview

This project demonstrates how deep learning and clustering techniques can be combined to understand and group large collections of face images.
The workflow includes:

Loading & preprocessing the CelebA dataset

Extracting 2048-dimensional deep facial embeddings using ResNet50

Reducing feature dimensionality using PCA

Applying K-Means clustering

Visualizing clusters using t-SNE

Evaluating clustering quality using internal & external metrics

This is extremely useful in real-world applications like photo organization, face retrieval, demographic analysis, and unsupervised face recognition.

📂 Dataset: CelebA
Property	Details
Images	202,599 aligned face images
Identities	10,177 unique people
Facial Attributes	40 binary facial attributes (Smiling, Young, Male, Blond Hair, etc.)
Facial Landmarks	5 points (eyes, nose, mouth corners)
Image Size	178 × 218
Used in this project	20,000 images (subset for computational efficiency)

CelebA is automatically downloaded using torchvision.datasets.CelebA.

⚙️ Methodology
1️⃣ Data Preprocessing

Images resized to 224×224

Converted to tensors

Normalized using ImageNet mean & std

Prepared using PyTorch transforms

2️⃣ Feature Extraction (ResNet50)

A pretrained ResNet50 model (without its classifier layer) is used

Each image produces a 2048-D embedding vector

These embeddings capture:

Expression

Age

Lighting

Gender

Hair style

3️⃣ Dimensionality Reduction (PCA)

PCA reduces embeddings from 2048 → 50 dimensions

Removes redundant information

Speeds up clustering

Improves cluster separability

4️⃣ Clustering (K-Means)

K-Means groups similar faces into K clusters

Cluster count selected via Elbow Method

Each feature vector assigned to nearest centroid

5️⃣ Visualization

t-SNE converts reduced embeddings → 2D

Clusters plotted with different colors

12 random images shown per cluster to inspect visual similarity

6️⃣ Evaluation Metrics
Internal Metrics
Metric	Meaning
Silhouette Score	Measures compactness & separation
Davies–Bouldin Index	Lower = better cluster separation
Calinski–Harabasz Score	Higher = better clustering
External Metrics

(using CelebA facial attributes)

Metric	Meaning
NMI	Compares clusters with ground-truth attributes
ARI	Similarity between predicted & true labels
Cluster Purity	% of attribute-consistent images in each cluster
📊 Evaluation Results (Achieved)
Evaluation Metric	Value
Silhouette Score	0.0388
Davies–Bouldin Index	2.8165
Calinski–Harabasz Score	720.65
Normalized Mutual Info	0.0301
Adjusted Rand Index	0.0126
Cluster Purity (Smiling)	0.6184
🧠 Cluster Interpretations
Cluster 0

Mostly male faces, short hair, no beard, formal appearance.

Cluster 1

Mostly female faces with long hair, makeup, lighter skin.

Cluster 2

Older males, some with glasses, beards, round faces.

Cluster 3

Young female faces, makeup, stylish/fashion look.

Cluster 4

Mixed gender, diverse poses, outdoor backgrounds.

🛠️ Tech Stack

Python

PyTorch

Torchvision

scikit-learn

matplotlib / seaborn

NumPy

Google Colab

🚀 How to Run This Project
1. Clone the repository
git clone https://github.com/yourusername/face-attribute-clustering.git
cd face-attribute-clustering

2. Install dependencies
pip install -r requirements.txt

3. Run the notebook

Open in Jupyter Notebook or Google Colab:

Face_Attribute_Analysis.ipynb


The dataset will auto-download when executed.

🔮 Future Improvements

To improve clustering metrics and accuracy:

Use face-specific models (FaceNet, ArcFace, VGGFace2)

Try UMAP or autoencoders for dimensionality reduction

Use HDBSCAN, GMM, or Deep Embedded Clustering (DEC)

Fine-tune ResNet on CelebA for more discriminative embeddings

Add attribute-based or contrastive learning

📌 Real-World Applications

Face recognition

Photo organization

Demographic analytics

Emotion detection

Surveillance

Digital marketing

Identity verification

Deepfake detection
