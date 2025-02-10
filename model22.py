import json
import numpy as np

# Define the file path
file_path = "embeddings/user_1_embedding.json"

# Load JSON data
try:
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Extract query embedding
    query_embedding = np.array(data.get("embedding", []))

    # Validate extracted embedding
    if query_embedding.size == 0:
        print("❌ Query embedding is empty! Check JSON structure.")
    else:
        print("✅ Extracted Query Embedding (first 5 values):", query_embedding[:5])
        print("🔹 Query Embedding Shape:", query_embedding.shape)

except Exception as e:
    print("❌ Error reading file:", e)
import json
import numpy as np
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# === Step 2: Load Product Data ===
df = pd.read_csv("product data - product data (2).csv")

# === Step 3: Convert Text & Image Embeddings from Strings to NumPy Arrays ===
def convert_embedding(x):
    try:
        return np.array(ast.literal_eval(x)) if isinstance(x, str) else np.zeros(512)
    except:
        return np.zeros(512)  # Return zero vector if conversion fails

df["text_embedding"] = df["text_embedding"].apply(convert_embedding)
df["image_embedding"] = df["image_embedding"].apply(convert_embedding)

# === Step 4: Normalize the Query & Product Embeddings ===
query_embedding = normalize(query_embedding.reshape(1, -1))[0]
df["text_embedding"] = df["text_embedding"].apply(lambda x: normalize(x.reshape(1, -1))[0])
df["image_embedding"] = df["image_embedding"].apply(lambda x: normalize(x.reshape(1, -1))[0])

# === Step 5: Compute Cosine Similarity ===
# Stack all product embeddings into numpy arrays
text_embeddings = np.stack(df["text_embedding"].values)
image_embeddings = np.stack(df["image_embedding"].values)

# Compute cosine similarity
text_similarities = cosine_similarity([query_embedding], text_embeddings)[0]
image_similarities = cosine_similarity([query_embedding], image_embeddings)[0]

# === Step 6: Hybrid Search (Weighted Combination of Text & Image Similarities) ===
text_weight = 0.6   # Adjust weights based on importance
image_weight = 0.4

df["similarity_score"] = (text_weight * text_similarities) + (image_weight * image_similarities)

# === Step 7: Sort Results & Display Top Recommendations ===
df = df.sort_values(by="similarity_score", ascending=False)

top_results = df[
    [
        "product_name",
        "price",
        "fabric",
        "pattern",
        "product_rating",
        "product_type",
        "occasion",
        "product_url",
        "image_url",
        "similarity_score",
    ]
].head(5)

print("\n🔹 **Top 5 Recommended Products:**")
print(top_results.to_string(index=False))
