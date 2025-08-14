# Qwen3 RAG System

A complete Retrieval-Augmented Generation (RAG) system using Qwen3 models, optimized for Google Colab T4 GPU with memory-efficient model loading/unloading.

## 🚀 Features

- **Multi-model RAG pipeline** with three specialized Qwen3 models:
  - `Qwen3-Embedding-0.6B` for document embeddings
  - `Qwen3-Reranker-0.6B` for document reranking  
  - `Qwen3-4B-Instruct-2507` for answer generation
- **Memory optimized** for Colab T4 (15GB VRAM) with automatic model loading/unloading
- **FAISS vector store** for efficient similarity search
- **Document processing** supports PDF and TXT files
- **Comprehensive logging** of retrieval and reranking results
- **Jupyter notebook** interface for interactive usage

## 📋 Requirements

```bash
pip install faiss-cpu PyPDF2 transformers torch numpy
```

## 🏗️ Project Structure

```
├── qwen_rag.py              # Main RAG system implementation
├── Qwen3_RAG_Notebook.ipynb # Jupyter notebook interface
├── Data/                    # Document folder (PDFs and TXT files)
├── vector_store.faiss       # FAISS index (auto-generated)
├── vector_store_docs.pkl    # Document metadata (auto-generated)
└── rag_retrieval_log.txt    # Detailed retrieval logs (auto-generated)
```

## 🚀 Quick Start

### Command Line Usage

1. **Prepare your documents**: Place PDF and TXT files in a `Data/` folder

2. **Run the RAG system**:
```bash
python qwen_rag.py
```

The system will:
- Process all documents in the `Data/` folder
- Build a FAISS vector store (cached for future runs)
- Answer the default query: "What is the difference between LoRA and QLoRA?"

### Jupyter Notebook Usage

Open `Qwen3_RAG_Notebook.ipynb` in Google Colab or Jupyter:

```python
!python qwen_rag.py
```

## 🔧 How It Works

### 1. Document Processing
- Extracts text from PDF files using PyPDF2
- Chunks documents into 512-token segments with 50-token overlap
- Processes both PDF and TXT files recursively

### 2. Vector Store Creation
- Generates embeddings using `Qwen3-Embedding-0.6B`
- Creates FAISS index with cosine similarity
- Caches vector store for subsequent runs

### 3. RAG Pipeline
1. **Retrieval**: Search top-15 similar documents using FAISS
2. **Reranking**: Rerank to top-3 using `Qwen3-Reranker-0.6B`
3. **Generation**: Generate answer using `Qwen3-4B-Instruct-2507`

### 4. Memory Management
- Automatic model loading/unloading between steps
- GPU memory monitoring and cleanup
- Optimized for 15GB VRAM constraint

## 📊 Memory Usage

| Stage | Model | GPU Memory |
|-------|-------|------------|
| Embedding | Qwen3-Embedding-0.6B | ~1.2GB |
| Reranking | Qwen3-Reranker-0.6B | ~1.2GB |
| Generation | Qwen3-4B-Instruct | ~8.1GB |

## 🔍 Customization

### Change the Query
Edit the `main()` function in `qwen_rag.py`:

```python
question = "Your custom question here"
```

### Adjust Retrieval Parameters
```python
# Number of candidates to retrieve
candidates = search_documents(question, index, documents, k=15)

# Number of documents to rerank
reranked = rerank_documents(question, candidates, k_rerank=3)
```

### Modify Generation Parameters
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,    # Adjust response length
    temperature=0.7,       # Control randomness
    top_p=0.9,            # Nucleus sampling
    do_sample=True
)
```

## 📝 Logging

The system generates detailed logs in `rag_retrieval_log.txt`:
- Document retrieval results with similarity scores
- Reranking results with reranking scores
- Top-3 selected documents for answer generation

## 🎯 Use Cases

- **Research assistance**: Query large document collections
- **Knowledge base QA**: Build internal knowledge systems
- **Document analysis**: Extract insights from PDF libraries
- **Educational tools**: Create interactive learning systems

## 🔧 Technical Details

- **Embedding dimension**: 1024 (Qwen3-Embedding-0.6B)
- **Chunk size**: 512 tokens with 50-token overlap
- **Similarity metric**: Cosine similarity (FAISS IndexFlatIP)
- **Precision**: FP16 on GPU, FP32 on CPU
- **Batch processing**: 4 documents per batch for embeddings

## 🚨 Limitations

- Optimized for Google Colab T4 (may need adjustments for other GPUs)
- English language focus (though Qwen3 supports multilingual)
- PDF extraction quality depends on document structure
- Maximum context length limited by model constraints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample documents
5. Submit a pull request

## 📄 License

This project is open source. Please check individual model licenses:
- [Qwen3 Models License](https://huggingface.co/Qwen)

## 🙏 Acknowledgments

- **Qwen Team** for the excellent Qwen3 model family
- **FAISS** for efficient vector search
- **Hugging Face** for model hosting and transformers library
