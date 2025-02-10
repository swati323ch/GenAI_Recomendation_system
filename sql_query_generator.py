import json
import os
from db import client
from get_user_data_agent import chatbot
def load_user_embedding(user_id):
    """
    Loads precomputed vector embeddings for a given user ID.

    Args:
        user_id (str): The ID of the user.

    Returns:
        list: The stored embedding vector or a default zero vector.
    """
    file_path = f"embeddings/user_{user_id}_embedding.json"

    if not os.path.exists(file_path):
        print(f"\n⚠️ Debug: Embedding file not found for User {user_id}. Using a default zero vector.")
        return [0.0] * 512  # Default to zero vector if missing

    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        print(f"\n✅ Debug: Successfully loaded embedding for User {user_id} from {file_path}")
        return data.get("embedding", [0.0] * 512)
    except Exception as e:
        print(f"\n❌ Debug: Error loading embedding for User {user_id} - {e}")
        return [0.0] * 512  # Fallback to zero vector

def get_results(user_id, user_preferences):
    """
    Fetches product recommendations based on user preferences and stored embeddings.

    Args:
        user_id (str): The user ID (to fetch their embedding).
        user_preferences (dict): The dictionary containing extracted user inputs.

    Returns:
        list[dict]: A list of product recommendations.
    """
    if not user_preferences:
        raise ValueError("❌ Debug: No user preferences provided.")

    print(f"\n🛍️ Debug: User Preferences: {user_preferences}")

    try:
        query = sql_generate(user_id, user_preferences)
        print(f"\n📝 Debug: Generated SQL Query:\n{query}\n")

        query_result = client.query(query)

        if not query_result or not query_result.result_rows:
            print("\n⚠️ Debug: Query executed but returned no results.")
            return []

        formatted_results = [dict(zip(query_result.column_names, row)) for row in query_result.result_rows]
        print(f"\n✅ Debug: Query Results: {formatted_results}")

        return formatted_results

    except Exception as e:
        print(f"\n❌ Debug: Database Query Failed: {e}")
        return []

def sql_generate(user_id, user_preferences) -> str:
    """
    Generates an SQL query using stored embeddings for hybrid search.

    Uses **precomputed** embeddings from `user_{user_id}_embedding.json`.

    Extracted fields:
      - product_type: Clothing type (e.g., dresses, shirts).
      - occasion: Use-case (e.g., casual, formal).
      - rating: Minimum product rating.
      - price: Maximum price.
      - pattern: Clothing pattern (e.g., floral, solid).
      - description: Query description for semantic similarity search.

    SQL Query assumes:
      - `getStart.product4_data_v2` table with:
        - product_type, occasion, rating, price, pattern, text_embedding

    Returns:
        str: The dynamically generated SQL query.
    """
    if not user_preferences:
        raise ValueError("❌ Debug: No user preferences provided for SQL generation.")

    # Load precomputed embedding for this user
    emb_query = load_user_embedding(user_id)

    sql = """
        SELECT 
            product_description, 
            product_rating, 
            occasion, 
            HybridSearch(
                'fusion_type=RRF', 
                'fusion_k=60'
            )(text_embedding, product_description, materialize({embedding}), 'BGLE') AS score 
        FROM 
            getStart.product4_data_v2
    """
    
    conditions = []

    # Extract filters
    product_type = user_preferences.get("product_type")
    occasion = user_preferences.get("occasion")
    rating = user_preferences.get("product_rating")
    price = user_preferences.get("price")
    pattern = user_preferences.get("pattern")

    sql = sql.format(embedding=emb_query)

    # Apply filters dynamically
    if product_type:
        conditions.append(f"LOWER(product_type) = '{product_type.lower()}'")
    if occasion:
        conditions.append(f"LOWER(occasion) = '{occasion.lower()}'")
    if rating:
        conditions.append(f"product_rating >= {rating}")
    if price:
        conditions.append(f"price <= {price}")
    if pattern:
        conditions.append(f"LOWER(pattern) = '{pattern.lower()}'")

    # Add conditions to the query
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    # Limit results
    sql += " ORDER BY score DESC LIMIT 10"

    print(f"\n🔎 Debug: Final SQL Query:\n{sql}")
    
    return sql
