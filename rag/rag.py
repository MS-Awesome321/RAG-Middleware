from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from semantic_cache import SemanticCache
import os
import numpy as np

# 1. Load a pretrained CrossEncoder model
se = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
ns = 'namespace1'
index_name = "toys-index"
index = pc.Index(index_name)
vector_dims = 384


class RAG:
    def __init__(self, top_k_after_rerank=5, top_k_initial=50, max_cache_len=10, cache_hit_threshold=0.5) -> None:
        self.cache = SemanticCache(vector_dims, maxlen=max_cache_len, threshold=cache_hit_threshold)
        self.tk1 = top_k_initial
        self.tk2 = top_k_after_rerank

    # Search Vector DB
    def search_db(self, query_vector):
        query_vector_list = np.squeeze(query_vector).tolist()
        results = index.query(
            namespace=ns,
            vector=query_vector_list,
            top_k=self.tk1,
            include_metadata=True
        )
        return results.matches

    # Rerank Results
    def rerank(self, query:str, matches:list) -> list[dict]:
        documents = [m['metadata']['text'] for m in matches if 'metadata' in m and 'text' in m['metadata']]
    
        if not documents:
            return []

        ranks = ce.rank(query=query, documents=documents, top_k=self.tk2, return_documents=True)
        return ranks

    # RAG Prompt Augmentation
    def augment_prompt(self, prompt):
        query_vector = se.encode(prompt)

        augmentation = self.cache.search(query_vector)
        if augmentation is not None:
            return f"{prompt} \nContext: \n{augmentation}"
        else:
            augmentation = ""

            matches = self.search_db(query_vector)
            ranks = self.rerank(prompt, matches)

            for rank in ranks:
                augmentation += f"- {rank['text']}\n\n"
            self.cache.append(query_vector, augmentation)

            return f"{prompt} \n\nContext: \n\n{augmentation}"


if __name__=='__main__':
    from time import time

    query = "Find me the best rocket kits for kids between 12-15 years old."
    rag = RAG()
    times = []

    start = time()
    print(rag.augment_prompt(query))
    end = time()
    times.append((end - start))
    
    start = time()
    query = "Find me the best lego sets for kids between 12-15 years old."
    print(rag.augment_prompt(query))
    end = time()
    times.append((end - start))

    start = time()
    query = "Find me the best rocket kits for adults."
    print(rag.augment_prompt(query))
    end = time()
    times.append((end - start))

    print(times)