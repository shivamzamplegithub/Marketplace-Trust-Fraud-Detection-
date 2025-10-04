import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# ---------------------- Load Model ----------------------
model = MultiTrustGNN(hidden_channels=32)
model.load_state_dict(torch.load("multi_trust_gnn.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()

# ---------------------- New Reviews Batch ----------------------
new_reviews = [
    {
        "text": "Perfect product, works like a charm!",
        "rating": 5,
        "verified": 1,
        "product_id": "p203",
        "ip_id": "ip33",
        "label": 1.0
    },
    {
        "text": "Terrible experience. Waste of money.",
        "rating": 3,
        "verified": 0,
        "product_id": "p220",
        "ip_id": "ip33",
        "label": 1.0
    },
    {
        "text": "Worth every penny!!!!",
        "rating": 5,
        "verified": 0,
        "product_id": "p207",
        "ip_id": "ip33",
        "label": 1.0
    },
    {
        "text": "Fantastic and most trusted",
        "rating": 3,
        "verified": 0,
        "product_id": "p230",
        "ip_id": "ip33",
        "label": 1.0
    }
]

# ---------------------- Save Pre-Finetune Scores ----------------------
model.eval()
with torch.no_grad():
    pre_scores = model(data.x_dict, data.edge_index_dict)
pre_ip_scores = pre_scores['ip'].clone()
pre_seller_scores = pre_scores['seller'].clone()

# ---------------------- Add Each Review ----------------------
added_reviews = []
new_review_ids = []

for i, r in enumerate(new_reviews):
    assert r["product_id"] in product_map and r["ip_id"] in ip_map, "❌ Product/IP not found."

    # Encode features
    new_bert = get_bert_embedding(r["text"]).reshape(1, -1)
    bert_with_new = np.vstack([bert_embeddings, new_bert])
    bert_with_new = MinMaxScaler().fit_transform(bert_with_new)
    new_bert_scaled = bert_with_new[-1]

    new_tfidf = tfidf.transform([r["text"]]).toarray()[0]
    new_sentiment = sia.polarity_scores(r["text"])['compound']
    new_meta = np.array([r["rating"], r["verified"]])
    new_feature = np.hstack([new_bert_scaled, new_tfidf, new_sentiment, new_meta])
    new_feature = torch.tensor(new_feature, dtype=torch.float).unsqueeze(0)

    # Indices
    new_idx = data['review'].x.size(0)
    product_idx = product_map[r["product_id"]]
    ip_idx = ip_map[r["ip_id"]]

    # Append feature & label
    data['review'].x = torch.cat([data['review'].x, new_feature], dim=0)
    data['review'].y = torch.cat([data['review'].y, torch.tensor([r["label"]], dtype=torch.float)], dim=0)
    data['review'].train_mask = torch.cat([data['review'].train_mask, torch.tensor([True])], dim=0)

    # Append edges
    data['review', 'written_for', 'product'].edge_index = torch.cat([
        data['review', 'written_for', 'product'].edge_index,
        torch.tensor([[new_idx], [product_idx]], dtype=torch.long)
    ], dim=1)
    data['review', 'sent_from', 'ip'].edge_index = torch.cat([
        data['review', 'sent_from', 'ip'].edge_index,
        torch.tensor([[new_idx], [ip_idx]], dtype=torch.long)
    ], dim=1)

    # Get seller
    p2s_edge = data['product', 'sold_by', 'seller'].edge_index
    matches = (p2s_edge[0] == product_idx).nonzero(as_tuple=True)[0]
    seller_idx = p2s_edge[1][matches[0]].item() if len(matches) > 0 else None

    new_id = f"new_review_{i+1}"
    new_review_ids.append(new_id)

    added_reviews.append({
        "review_idx": new_idx,
        "review_id": new_id,
        "ip_idx": ip_idx,
        "ip_id": r["ip_id"],
        "seller_idx": seller_idx,
        "label": r["label"],
        "text": r["text"]
    })

# ---------------------- Safe Loss Function ----------------------
def safe_loss(pred, label):
    min_len = min(pred.size(0), label.size(0))
    return loss_fn(pred[:min_len], label[:min_len])

# ---------------------- Fine-tune ----------------------
for epoch in range(5):
    model.train()
    out = model(data.x_dict, data.edge_index_dict)
    loss_review = safe_loss(out['review'], data['review'].y)
    loss_ip = safe_loss(out['ip'], data['ip'].y)
    loss_seller = safe_loss(out['seller'], data['seller'].y)
    loss = loss_review + loss_ip + loss_seller
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# ---------------------- Post-Finetune Scores ----------------------
model.eval()
with torch.no_grad():
    final_scores = model(data.x_dict, data.edge_index_dict)

# ---------------------- Reporting Impact ----------------------
print("\n📊 Trust Score Impact After Adding Multiple Reviews:")

for i, r in enumerate(added_reviews):
    ip_before = pre_ip_scores[r['ip_idx']].item()
    ip_after = final_scores['ip'][r['ip_idx']].item()
    print(f"\n🆕 Review {i+1}: {'REAL' if r['label'] == 1.0 else 'FAKE'}")
    print(f"📝 Text: {r['text']}")
    print(f"🔸 IP Score ({r['ip_id']}): {ip_before:.4f} → {ip_after:.4f}")
    if r['seller_idx'] is not None:
        seller_before = pre_seller_scores[r['seller_idx']].item()
        seller_after = final_scores['seller'][r['seller_idx']].item()
        print(f"🔸 Seller Score: {seller_before:.4f} → {seller_after:.4f}")
    else:
        print("⚠️ No seller linked.")

# ---------------------- Export Trust Scores ----------------------

# 1. Reviews (original + new)
all_review_ids = list(review_map.keys()) + new_review_ids
review_scores = final_scores['review'].cpu().numpy()
review_df = pd.DataFrame({
    "review_id": all_review_ids[:len(review_scores)],
    "trust_score": review_scores[:len(all_review_ids)]
})
review_df.to_csv("trust_scores_all_reviews.csv", index=False)

# 2. IPs
ip_ids = list(ip_map.keys())
ip_scores = final_scores['ip'].cpu().numpy()
ip_df = pd.DataFrame({
    "ip_id": ip_ids[:len(ip_scores)],
    "trust_score": ip_scores[:len(ip_ids)]
})
ip_df.to_csv("trust_scores_all_ips.csv", index=False)

# 3. Sellers
seller_ids = list(seller_map.keys())
seller_scores = final_scores['seller'].cpu().numpy()
seller_df = pd.DataFrame({
    "seller_id": seller_ids[:len(seller_scores)],
    "trust_score": seller_scores[:len(seller_ids)]
})
seller_df.to_csv("trust_scores_all_sellers.csv", index=False)

print("\n✅ CSV files saved:")
print("📁 trust_scores_all_reviews.csv")
print("📁 trust_scores_all_ips.csv")
print("📁 trust_scores_all_sellers.csv")