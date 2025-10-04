# ---------- 1. Load CSVs ----------
df_reviews = pd.read_csv("/content/review_wl.csv")
df_products = pd.read_csv("/content/product_wl.csv")
df_sellers = pd.read_csv("/content/seller_wl.csv")
df_ips = pd.read_csv("/content/ip_wl.csv")

# ---------- 2. Fix IP Issues ----------
df_ips = df_ips.drop_duplicates(subset='ip_id', keep='first').reset_index(drop=True)
missing_ips = set(df_reviews['ip_id']) - set(df_ips['ip_id'])
for ip in missing_ips:
    df_ips = pd.concat([df_ips, pd.DataFrame([{
        'ip_id': ip,
        'frequency': 1,
        'vpn_flag': 0,
        'label': 0
    }])], ignore_index=True)

# ---------- 3. Create ID Mappings ----------
review_map = {rid: i for i, rid in enumerate(df_reviews['review_id'])}
product_map = {pid: i for i, pid in enumerate(df_products['product_id'])}
seller_map = {sid: i for i, sid in enumerate(df_sellers['seller_id'])}
ip_map = {iid: i for i, iid in enumerate(df_ips['ip_id'])}

# ---------- 4. Feature Extraction ----------
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from torch_geometric.data import HeteroData

nltk.download('vader_lexicon', quiet=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

def get_bert_embedding(text):
    inputs = tokenizer(str(text), return_tensors='pt', truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

bert_embeddings = np.vstack([get_bert_embedding(t) for t in df_reviews['text']])
bert_embeddings = MinMaxScaler().fit_transform(bert_embeddings)
tfidf = TfidfVectorizer(max_features=28)
tfidf_matrix = tfidf.fit_transform(df_reviews['text'].astype(str)).toarray()
sia = SentimentIntensityAnalyzer()
sentiment_scores = np.array([sia.polarity_scores(str(t))['compound'] for t in df_reviews['text']]).reshape(-1, 1)
meta_features = df_reviews[['rating', 'verified']].values
review_features = np.hstack([bert_embeddings, tfidf_matrix, sentiment_scores, meta_features])

# ---------- 5. Other Features ----------
product_features = df_products[['price', 'return_rate']].values
seller_features = df_sellers[['avg_rating', 'flags', 'account_age']].values
ip_features = df_ips[['frequency', 'vpn_flag']].values

review_x = torch.tensor(review_features, dtype=torch.float)
product_x = torch.tensor(product_features, dtype=torch.float)
seller_x = torch.tensor(seller_features, dtype=torch.float)
ip_x = torch.tensor(ip_features, dtype=torch.float)

# ---------- 6. Edge Construction ----------
from sklearn.metrics.pairwise import cosine_similarity

def build_edge_index(src_ids, dst_ids, src_map, dst_map):
    src_idx, dst_idx = [], []
    for s, d in zip(src_ids, dst_ids):
        if s in src_map and d in dst_map:
            src_idx.append(src_map[s])
            dst_idx.append(dst_map[d])
    return torch.tensor([src_idx, dst_idx], dtype=torch.long)

edge_r2p = build_edge_index(df_reviews['review_id'], df_reviews['product_id'], review_map, product_map)
edge_p2s = build_edge_index(df_products['product_id'], df_products['seller_id'], product_map, seller_map)
edge_r2ip = build_edge_index(df_reviews['review_id'], df_reviews['ip_id'], review_map, ip_map)
sim_matrix = cosine_similarity(tfidf_matrix)
edges = np.argwhere(sim_matrix > 0.8)
edges = edges[edges[:, 0] != edges[:, 1]]
edge_sim = torch.tensor(edges.T, dtype=torch.long) if len(edges) > 0 else torch.empty((2, 0), dtype=torch.long)

# ---------- 7. HeteroData with Labels ----------
data = HeteroData()
data['review'].x = review_x
data['product'].x = product_x
data['seller'].x = seller_x
data['ip'].x = ip_x

data['review', 'written_for', 'product'].edge_index = edge_r2p
data['product', 'sold_by', 'seller'].edge_index = edge_p2s
data['review', 'sent_from', 'ip'].edge_index = edge_r2ip
data['review', 'similar_to', 'review'].edge_index = edge_sim

data['review'].y = torch.tensor(df_reviews['label'].fillna(0).values, dtype=torch.float)
data['ip'].y = torch.tensor(df_ips['label'].fillna(0).values, dtype=torch.float)
data['seller'].y = torch.tensor(df_sellers['label'].fillna(0).values, dtype=torch.float)

data['review'].train_mask = ~torch.isnan(torch.tensor(df_reviews['label'].values)).bool()
data['ip'].train_mask = ~torch.isnan(torch.tensor(df_ips['label'].values)).bool()
data['seller'].train_mask = ~torch.isnan(torch.tensor(df_sellers['label'].values)).bool()

# ---------- 8. GNN ----------
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv

class LinearWrapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    def forward(self, x, edge_index=None):
        if isinstance(x, tuple):
            x = x[0]
        return self.linear(x)

class MultiTrustGNN(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('review', 'written_for', 'product'): LinearWrapper(review_x.shape[1], hidden_channels),
            ('product', 'sold_by', 'seller'): LinearWrapper(product_x.shape[1], hidden_channels),
            ('review', 'sent_from', 'ip'): LinearWrapper(review_x.shape[1], hidden_channels),
            ('review', 'similar_to', 'review'): LinearWrapper(review_x.shape[1], hidden_channels),
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('review', 'written_for', 'product'): LinearWrapper(hidden_channels, hidden_channels),
            ('product', 'sold_by', 'seller'): LinearWrapper(hidden_channels, hidden_channels),
            ('review', 'sent_from', 'ip'): LinearWrapper(hidden_channels, hidden_channels),
            ('review', 'similar_to', 'review'): LinearWrapper(hidden_channels, hidden_channels),
        }, aggr='sum')
        self.lin_review = nn.Linear(hidden_channels, 1)
        self.lin_ip = nn.Linear(hidden_channels, 1)
        self.lin_seller = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return {
            'review': torch.sigmoid(self.lin_review(x_dict['review']).squeeze()),
            'ip': torch.sigmoid(self.lin_ip(x_dict['ip']).squeeze()),
            'seller': torch.sigmoid(self.lin_seller(x_dict['seller']).squeeze())
        }

# ---------- 9. Training ----------
model = MultiTrustGNN(hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

for epoch in range(50):
    model.train()
    out = model(data.x_dict, data.edge_index_dict)
    loss_review = loss_fn(out['review'][data['review'].train_mask], data['review'].y[data['review'].train_mask])
    loss_ip = loss_fn(out['ip'][data['ip'].train_mask], data['ip'].y[data['ip'].train_mask])
    loss_seller = loss_fn(out['seller'][data['seller'].train_mask], data['seller'].y[data['seller'].train_mask])
    loss = loss_review + loss_ip + loss_seller
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Review: {loss_review.item():.4f} | IP: {loss_ip.item():.4f} | Seller: {loss_seller.item():.4f}")

# ---------- Save Trained Model ----------
torch.save(model.state_dict(), "multi_trust_gnn.pth")
print("✅ Model saved as multi_trust_gnn.pth")


# ---------- 10. Save Trust Scores ----------
model.eval()
with torch.no_grad():
    scores = model(data.x_dict, data.edge_index_dict)
    df_reviews['trust_score'] = scores['review'].cpu().numpy()
    df_reviews[['review_id', 'trust_score']].to_csv('trust_scores_reviews.csv', index=False)
    ip_ids = list(ip_map.keys())
    ip_scores = scores['ip'].cpu().numpy()
    ip_min_len = min(len(ip_ids), len(ip_scores))
    pd.DataFrame({'ip_id': ip_ids[:ip_min_len], 'trust_score': ip_scores[:ip_min_len]}).to_csv('trust_scores_ips.csv', index=False)
    seller_ids = list(seller_map.keys())
    seller_scores = scores['seller'].cpu().numpy()
    seller_min_len = min(len(seller_ids), len(seller_scores))
    pd.DataFrame({'seller_id': seller_ids[:seller_min_len], 'trust_score': seller_scores[:seller_min_len]}).to_csv('trust_scores_sellers.csv', index=False)