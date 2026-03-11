# RAG-Middleware



A high-performance RAG (Retrieval-Augmented Generation) orchestration layer designed to reduce LLM latency and improve response relevance. This project focuses on the "Middle" of the stack: implementing **Semantic Caching** to skip redundant DB lookups and **Cross-Encoder Reranking** to ensure the LLM receives only the most pertinent context.

## 🚀 Features

* **Semantic Cache:** Uses vector similarity to store and retrieve previously answered queries, significantly reducing API costs and latency.
* **Two-Stage Retrieval:**
    1.  **Initial Retrieval:** Fast K-Nearest Neighbors search via **Pinecone**.
    2.  **Reranking:** Precision scoring of top results using `cross-encoder/ms-marco-MiniLM-L6-v2`.
* **Streamlit Chat Interface:** A clean, reactive UI for interacting with the toy inventory database.
* **Local Embedding Generation:** Powered by `sentence-transformers` (`all-MiniLM-L6-v2`) for efficient vectorization.

## 🛠️ Architecture

1.  **User Query:** The user asks a question about a toy.
2.  **Cache Check:** The system checks the `SemanticCache`. If a similar query exists above the similarity threshold, it returns the cached context immediately.
3.  **Vector Search:** If a cache miss occurs, the query is embedded and sent to **Pinecone** to retrieve the top 50 candidates.
4.  **Reranking:** The candidates are re-scored by the **Cross-Encoder** to identify the top 5 most relevant documents.
5.  **Augmentation:** The final refined context is injected into the prompt and sent to the LLM.

---

## 📋 Prerequisites

* Python 3.10+
* Pinecone API Key

## 🔧 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/RAG-Middleware.git](https://github.com/your-username/RAG-Middleware.git)
    cd RAG-Middleware
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables:**
    ```bash
    export PINECONE_API_KEY='your-api-key-here'
    ```

## 💻 Usage

To launch the Streamlit application:
```bash
cd app
streamlit run app.py