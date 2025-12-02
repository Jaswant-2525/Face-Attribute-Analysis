📌 **Face Attribute-Based Image Clustering using ResNet50 & K-Means**

This project performs unsupervised image clustering on the CelebA face dataset using deep feature extraction and machine learning.
A pretrained ResNet50 model is used to extract facial embeddings, followed by PCA for dimensionality reduction and K-Means for clustering.
The aim is to group visually similar faces based on attributes such as gender, age, expression, hairstyle, presence of glasses, etc.

🔍 **Project Overview**

This project demonstrates how deep learning and clustering techniques can be combined to understand and group large collections of face images.
The workflow includes:
1. Loading & preprocessing the CelebA dataset
2. Extracting 2048-dimensional deep facial embeddings using ResNet50
3. Reducing feature dimensionality using PCA
4. Applying K-Means clustering
5. Visualizing clusters using t-SNE
6. Evaluating clustering quality using internal & external metrics
7. This is extremely useful in real-world applications like photo organization, face retrieval, demographic analysis, and unsupervised face recognition.

📂 **Dataset: CelebA**
1. Property	Details
2. Images	202,599 aligned face images
3. Identities	10,177 unique people
4. Facial Attributes	40 binary facial attributes (Smiling, Young, Male, Blond Hair, etc.)
5. Facial Landmarks	5 points (eyes, nose, mouth corners)
6. Image Size	178 × 218
7. Used in this project	20,000 images (subset for computational efficiency)

CelebA is automatically downloaded using torchvision.datasets.CelebA.

⚙️ **Methodology**

1️⃣ Data Preprocessing
1. Images resized to 224×224
2. Converted to tensors
3. Normalized using ImageNet mean & std
4. Prepared using PyTorch transforms

2️⃣ Feature Extraction (ResNet50)
1.  pretrained ResNet50 model (without its classifier layer) is used
2. Each image produces a 2048-D embedding vector
3. These embeddings capture:
4. Expression
5. Age
6. Lighting
7. Gender
8. Hair style

3️⃣ Dimensionality Reduction (PCA)
1. PCA reduces embeddings from 2048 → 50 dimensions
2. Removes redundant information
3. Speeds up clustering
4. Improves cluster separability

4️⃣ Clustering (K-Means)
1. K-Means groups similar faces into K clusters
2. Cluster count selected via Elbow Method
3. Each feature vector assigned to nearest centroid

5️⃣ Visualization
1. t-SNE converts reduced embeddings → 2D
2. Clusters plotted with different colors
3. 12 random images shown per cluster to inspect visual similarity

6️⃣ Evaluation Metrics
1. Silhouette Score	Measures compactness & separation
2. Davies–Bouldin Index	Lower = better cluster separation
3. Calinski–Harabasz Score	Higher = better clustering

🧠 **Cluster Interpretations**

**Cluster 0**

Mostly male faces, short hair, no beard, formal appearance.

**Cluster 1**

Mostly female faces with long hair, makeup, lighter skin.

**Cluster 2**

Older males, some with glasses, beards, round faces.

**Cluster 3**

Young female faces, makeup, stylish/fashion look.

**Cluster 4**

Mixed gender, diverse poses, outdoor backgrounds.


🚀 **How to Run This Project**
1. **Clone the repository**
git clone https://github.com/yourusername/face-attribute-clustering.git
cd face-attribute-clustering

2. **Install dependencies**
pip install -r requirements.txt

3. **Run the notebook**

Open in Jupyter Notebook or Google Colab:
