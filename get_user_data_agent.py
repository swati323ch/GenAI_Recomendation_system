import json
import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import g4f  
from db import client  

# Required fields for fashion selection
REQUIRED_FIELDS = ["occasion", "product_type", "price", "product_rating"]

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_clip_embedding(text):
    """Generate vector embeddings using CLIP."""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**inputs)
    return text_embeddings.squeeze().tolist()  # Ensure embeddings remain Float32

def save_embedding(user_id, user_preferences):
    """Generate and save CLIP embeddings dynamically."""
    json_text = json.dumps(user_preferences, indent=4)
    json_embedding = generate_clip_embedding(json_text)
    os.makedirs("embeddings", exist_ok=True)
    file_path = f"embeddings/user_{user_id}_embedding.json"
    with open(file_path, "w") as file:
        json.dump({"embedding": json_embedding}, file)
    print(f"✅ CLIP Embedding saved for User {user_id} at {file_path}")


def load_user_embedding(user_id):
    file_path = f"embeddings/user_{user_id}_embedding.json"

    if not os.path.exists(file_path):
        print(f"\n Embedding file not found for User {user_id}. Using a default zero vector.")
        return [0.0] * 512 
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            embedding_vector = data.get("embedding", [0.0] * 512) 
        return embedding_vector
    except Exception as e:
        return [0.0] * 512

def sql_generate(user_id, user_preferences) -> str:
    if not user_preferences:
        raise ValueError("ERROR")

    # Load precomputed embedding for this user
    emb_query = load_user_embedding(user_id)
    emb_query_str = "[" + ", ".join(f"'{str(val)}'" for val in emb_query) + "]"
    sql = """
        SELECT 
            product_rating, 
            occasion, 
            HybridSearch(
                'fusion_type=RRF', 
                'fusion_k=60'
            )(
                text_embedding,
                materialize({emb_query_str}),
            ) AS score 
        FROM 
            getStart.product4_data_v2
    """
    conditions = []
 
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY score DESC LIMIT 5"
    return sql

def get_results(user_id, user_preferences):
    query = sql_generate(user_id, user_preferences)
    try:
        query_result = client.query(query)
        if not query_result or not query_result.result_rows:
            return []
        formatted_results = [dict(zip(query_result.column_names, row)) for row in query_result.result_rows]
        return formatted_results
    except Exception as e:
        print(f"Database Query Failed: {e}")
        return []

def chatbot():
    print("👗 Welcome to FashionBot! Let's find the perfect outfit for you.")
    user_query = input("👤 What are you looking for? ")

    while True:
        prompt = f"""
        Extract structured fashion preferences from this query and return JSON only.
        Query: "{user_query}"
        Keys:
        - "occasion": e.g., "formals", "casual".
        - "product_type": e.g., "Shirt", "Trousers".
        - "fabric": e.g., "cotton", "silk".
        - "pattern": e.g., "solid", "striped".
        - "product_rating": 1-5.
        - "price": [min, max].
        - "product_description".
        Output:
        """

        try:
            response = g4f.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.strip()
        except Exception as e:
            print(f"❌ Error: Failed to fetch response. Retrying...\n{str(e)}")
            continue

        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        try:
            user_preferences = json.loads(response_text)
            missing_fields = [field for field in REQUIRED_FIELDS if user_preferences.get(field) is None]

            if not missing_fields:
                print("✅ Here’s your fashion selection:")
                print(json.dumps(user_preferences, indent=4))
                
                user_id = input("🔹 Enter your User ID to save preferences: ")
                save_embedding(user_id, user_preferences)
                
                print("🔍 Fetching product recommendations... Please wait.")
                results = get_results(user_id, user_preferences)
                
                if results:
                    print("🎉 Here are your top recommendations:")
                    for idx, product in enumerate(results, 1):
                        print(f"{idx}. {product['product_description']} (⭐ {product['product_rating']}) - Occasion: {product['occasion']}")
                else:
                    print("😔 No matching products found. Try refining your search.")
                
                user_input = input("🔄 Do you want to search again? (yes/no): ").strip().lower()
                if user_input not in ["yes", "y"]:
                    print("👋 Exiting FashionBot. Have a stylish day! 👗✨")
                    break
        except json.JSONDecodeError:
            print("❌ Error: Model did not return valid JSON. Retrying...")
            continue

chatbot()
