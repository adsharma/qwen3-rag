#!/usr/bin/env python3
"""
Simple Qwen3 RAG System - Memory Optimized for Colab T4
Run with: python simple_qwen_rag.py
"""

import os
import torch
import numpy as np
import faiss
import pickle
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import PyPDF2
import litellm
import dotenv
import argparse
import requests
import json

# Global variables

# litellm._turn_on_debug()
dotenv.load_dotenv()
api_base = os.getenv("API_BASE")

device = "cuda" if torch.cuda.is_available() else "cpu"

args = None


def print_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"GPU Memory: {allocated:.1f}GB")


def clear_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=800, overlap=100):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def process_documents(data_folder):
    """Process all documents in folder"""
    documents = []
    data_path = Path(data_folder)

    for file_path in data_path.glob("*"):
        if file_path.suffix.lower() in [".pdf", ".txt"]:
            print(f"Processing {file_path.name}...")

            if file_path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                documents.append(
                    {"content": chunk, "source": file_path.name, "chunk_id": i}
                )

    print(f"Processed {len(documents)} document chunks")
    return documents


def generate_embeddings(documents):
    """Generate embeddings for documents using Ollama"""
    print("Generating embeddings with LiteLLM...")

    embeddings = []
    batch_size = 4

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        texts = [
            f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {doc['content']}"
            for doc in batch_docs
        ]

        # Generate embeddings using LiteLLM
        batch_embeddings = []
        for text in texts:
            response = litellm.embedding(
                model=args.embedding_model, input=text, api_base=api_base
            )
            batch_embeddings.append(response["data"][0]["embedding"])

        embeddings.extend(batch_embeddings)

    print("Embeddings generated!")
    return np.array(embeddings)


def create_vector_store(documents, embeddings, save_path="vector_store"):
    """Create FAISS vector store"""
    print("Creating vector store...")

    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save
    faiss.write_index(index, f"{save_path}.faiss")
    with open(f"{save_path}_docs.pkl", "wb") as f:
        pickle.dump(documents, f)

    print(f"Vector store saved to {save_path}")
    return index, documents


def load_vector_store(save_path="vector_store"):
    """Load existing vector store"""
    if os.path.exists(f"{save_path}.faiss"):
        print(f"Loading existing vector store from {save_path}")
        index = faiss.read_index(f"{save_path}.faiss")
        with open(f"{save_path}_docs.pkl", "rb") as f:
            documents = pickle.load(f)
        return index, documents
    return None, None


def search_documents(query, index, documents, k=15):
    """Search for relevant documents using Ollama embeddings"""
    print("Searching documents...")

    # Generate query embedding using LiteLLM
    query_text = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"
    response = litellm.embedding(
        model=args.embedding_model, input=query_text, api_base=api_base
    )
    query_embedding = (
        np.array(response["data"][0]["embedding"]).astype("float32").reshape(1, -1)
    )

    print("Query embedding generated")

    # Search
    similarities, indices = index.search(query_embedding, k)

    results = []
    for similarity, idx in zip(similarities[0], indices[0]):
        if idx < len(documents):
            results.append((documents[idx], float(similarity)))

    # Log retrieved documents
    with open("rag_retrieval_log.txt", "w", encoding="utf-8") as f:
        f.write(f"=== DOCUMENT RETRIEVAL LOG ===\n")
        f.write(f"Query: {query}\n")
        f.write(f"Retrieved {len(results)} documents\n\n")

        for i, (doc, similarity) in enumerate(results):
            f.write(f"--- Document {i+1} ---\n")
            f.write(f"Source: {doc['source']}\n")
            f.write(f"Chunk ID: {doc['chunk_id']}\n")
            f.write(f"Similarity Score: {similarity:.4f}\n")
            f.write(f"Content: {doc['content'][:200]}...\n\n")

    return results


def load_reranker_model():
    """Load reranker model"""
    print("Loading reranker model...")
    clear_memory()

    tokenizer = AutoTokenizer.from_pretrained(args.reranker_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.reranker_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)

    print("Reranker model loaded!")
    print_memory()
    return tokenizer, model


def rerank_documents_vllm(query, candidates, vllm_url, k_rerank=3):
    """Rerank documents using vLLM reranker API"""
    print("Reranking documents with vLLM...")

    # Prepare reranking data
    rerank_data = {
        "query": query,
        "documents": [doc["content"] for doc, _ in candidates],
    }

    try:
        # Make request to vLLM reranker endpoint
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{vllm_url}/rerank", headers=headers, json=rerank_data, timeout=30
        )
        response.raise_for_status()

        result = response.json()
        scores = result.get("scores", [])

        if len(scores) != len(candidates):
            raise ValueError(f"Expected {len(candidates)} scores, got {len(scores)}")

        print(f"Received scores from vLLM: {scores}")

    except Exception as e:
        print(f"Error calling vLLM reranker: {e}")
        print("Falling back to similarity scores...")
        # Fallback to original similarity scores
        scores = [score for _, score in candidates]

    # Sort by score
    documents = [doc for doc, _ in candidates]
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # Log reranking results
    with open("rag_retrieval_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n=== vLLM RERANKING RESULTS ===\n")
        f.write(f"vLLM URL: {vllm_url}\n")
        f.write(f"All {len(doc_scores)} documents with reranking scores:\n\n")

        for i, (doc, score) in enumerate(doc_scores):
            f.write(f"--- Reranked Document {i+1} ---\n")
            f.write(f"Source: {doc['source']}\n")
            f.write(f"Chunk ID: {doc['chunk_id']}\n")
            f.write(f"Reranking Score: {score:.4f}\n")
            f.write(f"Content: {doc['content'][:200]}...\n\n")

        f.write(f"=== TOP {k_rerank} SELECTED DOCUMENTS ===\n")
        for i, (doc, score) in enumerate(doc_scores[:k_rerank]):
            f.write(f"--- Selected Document {i+1} ---\n")
            f.write(f"Source: {doc['source']}\n")
            f.write(f"Chunk ID: {doc['chunk_id']}\n")
            f.write(f"Reranking Score: {score:.4f}\n")
            f.write(f"Full Content: {doc['content']}\n\n")

    return doc_scores[:k_rerank]


def rerank_documents(query, candidates, k_rerank=3):
    """Rerank documents using reranker model"""
    print("Reranking documents...")
    tokenizer, model = load_reranker_model()

    # Prepare inputs
    pairs = []
    for doc, _ in candidates:
        pair = f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {doc['content']}"
        pairs.append(pair)

    # Tokenize
    inputs = tokenizer(
        pairs, padding=True, truncation=True, max_length=8192, return_tensors="pt"
    ).to(device)

    # Get scores
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]

        # Get yes/no token scores
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")

        true_scores = logits[:, token_true_id]
        false_scores = logits[:, token_false_id]

        batch_scores = torch.stack([false_scores, true_scores], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()

    # Unload reranker model
    del tokenizer, model
    clear_memory()
    print(scores)
    print("Reranker model unloaded")

    # Sort by score
    documents = [doc for doc, _ in candidates]
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # Log reranking results
    with open("rag_retrieval_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n=== RERANKING RESULTS ===\n")
        f.write(f"All {len(doc_scores)} documents with reranking scores:\n\n")

        for i, (doc, score) in enumerate(doc_scores):
            f.write(f"--- Reranked Document {i+1} ---\n")
            f.write(f"Source: {doc['source']}\n")
            f.write(f"Chunk ID: {doc['chunk_id']}\n")
            f.write(f"Reranking Score: {score:.4f}\n")
            f.write(f"Content: {doc['content'][:200]}...\n\n")

        f.write(f"=== TOP {k_rerank} SELECTED DOCUMENTS ===\n")
        for i, (doc, score) in enumerate(doc_scores[:k_rerank]):
            f.write(f"--- Selected Document {i+1} ---\n")
            f.write(f"Source: {doc['source']}\n")
            f.write(f"Chunk ID: {doc['chunk_id']}\n")
            f.write(f"Reranking Score: {score:.4f}\n")
            f.write(f"Full Content: {doc['content']}\n\n")

    return doc_scores[:k_rerank]


def generate_answer_with_ollama(query, context_docs):
    """Generate answer using Ollama qwen3:0.6b model"""
    print("Generating answer with LiteLLM...")

    # Prepare context
    context_text = "\n\n".join(
        [
            f"Document {i+1} from {doc['source']}:\n{doc['content']}"
            for i, (doc, _) in enumerate(context_docs)
        ]
    )

    # Create prompt
    prompt = f"""Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""

    # Generate response using LiteLLM
    response = litellm.completion(
        model=args.generation_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        api_base=api_base,
    )

    # Extract the response text
    answer_text = response["choices"][0]["message"]["content"].strip()

    # Add sources
    sources = ", ".join([doc["source"] for doc, _ in context_docs])
    return f"{answer_text}\n\nSources: {sources}"


def generate_answer(query, context_docs):
    """Generate answer using instruct model"""
    return generate_answer_with_ollama(query, context_docs)


def main():
    """Main function"""
    global args
    parser = argparse.ArgumentParser(description="Qwen3 RAG System")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="ollama/dengcao/Qwen3-Embedding-0.6B:Q8_0",
        help="Embedding model name",
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
        help="Reranker model name",
    )
    parser.add_argument(
        "--generation_model",
        type=str,
        default="ollama/qwen3:0.6b",
        help="Generation model name",
    )
    parser.add_argument(
        "--data_folder", type=str, default="Data", help="Folder containing documents"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are popular PEFT techniques?",
        help="Query string",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Top K documents to retrieve"
    )
    parser.add_argument(
        "--top_k_rerank", type=int, default=3, help="Top K documents to rerank"
    )
    parser.add_argument(
        "--vllm-reranker",
        type=str,
        default=None,
        help="vLLM reranker URL (e.g., http://localhost:8000). When specified, bypasses local reranker.",
    )
    args = parser.parse_args()

    print("=== Simple Qwen3 RAG System ===")
    print_memory()

    # Check if vector store exists
    index, documents = load_vector_store()

    if index is None:
        # Build vector store
        print(f"\nBuilding vector store from {args.data_folder} folder...")
        documents = process_documents(args.data_folder)
        embeddings = generate_embeddings(documents)
        index, documents = create_vector_store(documents, embeddings)

    # Query
    question = args.query
    print(f"\nQuery: {question}")

    # Step 1: Search
    candidates = search_documents(question, index, documents, k=args.top_k)
    print(f"Found {len(candidates)} candidates")

    # Step 2: Rerank
    if args.vllm_reranker:
        reranked = rerank_documents_vllm(
            question, candidates, args.vllm_reranker, k_rerank=args.top_k_rerank
        )
        print(f"Reranked with vLLM to top {len(reranked)} documents")
    else:
        reranked = rerank_documents(question, candidates, k_rerank=args.top_k_rerank)
        print(f"Reranked with local model to top {len(reranked)} documents")

    # Step 3: Generate answer
    answer = generate_answer(question, reranked)

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")

    print("\n=== RAG Query Completed ===")
    print_memory()


if __name__ == "__main__":
    main()
