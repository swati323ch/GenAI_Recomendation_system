# OutStyle-Task-Submission
Here is a refined **README** file without the code snippets:  

---

# **Fashion Style Recommender System**  

## **Overview**  
The **Fashion Style Recommender System** is a personalized fashion recommendation engine that leverages **vector embeddings**, **semantic search**, and **content-based filtering** to enhance user experience. It integrates a **conversational agent** to dynamically generate user queries and provide **hybrid recommendations** based on product descriptions, images, price, and rating.  

## **Features**  
- Conversational agent for interactive query generation.  
- Vector-based search using text and image embeddings.  
- Hybrid recommendation model combining **semantic search** and **content-based filtering**.  
- Efficient database storage with **MyScale (ClickHouse)** for fast retrieval.  
- User-friendly interface for displaying recommendations.  

## **Project Architecture**  
The system follows a structured pipeline consisting of:  

- **Data Collection** – Scraping product details, including descriptions, images, price, and ratings.  
- **Creating Vector Embeddings** – Generating embeddings for product descriptions using an NLP model and for images using a CNN model.  
- **Setting Up MyScale Database** – Storing text and image embeddings in a MyScale (ClickHouse) vector database.  
- **Indexing for Faster Search** – Creating indexes for product descriptions and image embeddings for efficient retrieval.  
- **Building a Conversational Agent** – Implementing a LangGraph-based chatbot using a local LLM (Llama.cpp) to dynamically generate user queries.  
- **Processing User Query** – Capturing user preferences and converting the query into an appropriate format while generating query embeddings.  
- **Performing Semantic Search** – Matching user query embeddings with product description and image embeddings using cosine similarity, then combining results into a hybrid search model.  
- **Applying Content-Based Filtering** – Filtering recommendations based on price and rating extracted from the user query.  
- **Combining and Displaying Results** – Merging results from semantic search and content-based filtering, then displaying personalized fashion recommendations through the UI.  

## **Installation & Setup**  
- Clone the repository and install the required dependencies.  
- Set up the **MyScale (ClickHouse) database** for storing product embeddings.  
- Run the **conversational agent** to gather user inputs.  
- Process the user query and perform **semantic search** combined with **content-based filtering**.  
- Display recommendations through the **UI**.  

## **Usage**  
1. Users interact with the chatbot to specify fashion preferences.  
2. The system processes the query and retrieves relevant recommendations.  
3. The recommendations are displayed through the user interface.  

## **Technologies Used**  
- **Python** – Backend processing and integration.  
- **LangGraph & LangChain** – Conversational agent implementation.  
- **Llama.cpp** – Local LLM for chatbot responses.  
- **MyScale (ClickHouse)** – Vector database for storing embeddings.  
- **NLP & CNN Models** – For generating text and image embeddings.  
- **Cosine Similarity** – Used for performing semantic search.  

## Results
Got the vector accuracy of 90 percent
![image](https://github.com/user-attachments/assets/685c9928-6c65-44e1-97bd-b2f262dacd44)

