# 🛡️ Amazon Hackathon Season 5 – Trust Score Prediction using Heterogeneous Graph Neural Networks

🔗 GitHub Repository: https://github.com/Ashutosh6294389/Amazon-Hackathon-Season-5

This repository implements a Heterogeneous Graph Neural Network (MultiTrustGNN) designed to predict TrustScores for reviews, IP addresses, and sellers on an e-commerce platform. The model uses a heterogeneous graph structure and node-level features such as BERT embeddings, TF-IDF vectors, sentiment scores, and metadata for effective relational reasoning.

---

## 📁 Repository Structure

Amazon-Hackathon-Season-5/
├── Model.py                     # GNN model architecture using PyTorch Geometric  
├── testing_file.py              # Inference script for adding new review nodes and predicting trust scores  
├── Training Dataset/            # Input CSVs: review_wl.csv, ip_wl.csv, seller_wl.csv, product_wl.csv  
├── multi_trust_gnn.pth          # Pretrained model weights  
├── trust_scores_all_reviews.csv # Output: trust scores of all review nodes  
├── trust_scores_all_ips.csv     # Output: trust scores of all IP nodes  
├── trust_scores_all_sellers.csv # Output: trust scores of all seller nodes  
└── README.md                    # Instructions and documentation  

---

## ✅ Installation Requirements

Run the following to install all required packages:

pip install torch torchvision torchaudio  
pip install torch-geometric  
pip install transformers  
pip install scikit-learn  
pip install nltk  

Also download the VADER sentiment lexicon:

import nltk  
nltk.download('vader_lexicon')  

---

## 🚀 Running the Model in Google Colab or Locally

Step 1: Clone the Repository

git clone https://github.com/Ashutosh6294389/Amazon-Hackathon-Season-5.git  
cd Amazon-Hackathon-Season-5  

Step 2: Prepare Model Weights

Ensure multi_trust_gnn.pth is present in the root directory. If you're using Google Colab:

from google.colab import files  
files.upload()  # Upload multi_trust_gnn.pth  

Step 3: Run the Inference Pipeline

Execute the testing script to:

- Load the pretrained GNN model  
- Add new review nodes dynamically  
- Fine-tune on known review labels (if provided)  
- Predict updated TrustScores for reviews, sellers, and IPs  
- Save the scores to CSV files for inspection  

python testing_file.py  

---

## 📤 Outputs

After running testing_file.py, the following CSV files will be generated/updated:

- trust_scores_all_reviews.csv – Trust scores for all reviews (existing + new)  
- trust_scores_all_ips.csv – Trust scores for all IP addresses  
- trust_scores_all_sellers.csv – Trust scores for all sellers  

Each score ranges from 0 (trustworthy) to 1 (likely fraudulent).

---

📌 Model Highlights

- HeteroGNN trained on review–product–seller–IP graphs  
- Node features include textual embeddings, ratings, and behavioral stats  
- Multi-hop message passing captures relational fraud patterns  
- Useful for detecting review bots, IP abuse, and seller fraud rings  
