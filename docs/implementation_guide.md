# Implementation Guide

This guide provides module-by-module explanations of the RAG pipeline implementation.

## Utility Modules

### Configuration Management (`utils/config.py`)

The `Config` class provides centralized configuration management:

```python
from utils.config import get_config

config = get_config()
model_name = config.get("llm.model_name", "gpt-4")
api_key = config.get_required("llm.api_key")  # Raises if missing
```

**Key Features:**
- Loads from environment variables, .env files, and YAML/JSON
- Dot notation for nested keys
- Type-safe accessors
- Singleton pattern for global config

### Logging (`utils/logging_utils.py`)

Structured logging with JSON output:

```python
from utils.logging_utils import get_logger, add_context

logger = get_logger(__name__)
add_context(request_id="req123", user_id="user456")
logger.info("Processing document", document_id="doc123")
```

**Key Features:**
- JSON-formatted logs for production
- Context variables for tracing
- Standard log fields

### Validation (`utils/validation.py`)

Input validation for documents, chunks, and queries:

```python
from utils.validation import validate_document, validate_query

doc = {"text": "content", "metadata": {}}
validate_document(doc)  # Raises ValidationError if invalid

validate_query("What is RAG?")  # Validates query format
```

### LLM Utilities (`utils/llm_utils.py`)

Unified interface for LLM calls:

```python
from utils.llm_utils import call_llm, build_rag_prompt

response = call_llm(
    prompt="What is RAG?",
    model="gpt-4",
    provider="openai"
)

prompt = build_rag_prompt(
    query="What is RAG?",
    context="Retrieved context here..."
)
```

**Supported Providers:**
- OpenAI
- AWS Bedrock
- Local models (placeholder)

### Embeddings (`utils/embeddings.py`)

Embedding generation with batching:

```python
from utils.embeddings import generate_embeddings, batch_embeddings

embedding = generate_embeddings("text", model="text-embedding-ada-002")
embeddings = batch_embeddings(["text1", "text2"], batch_size=100)
```

**Supported Providers:**
- OpenAI
- Sentence Transformers
- AWS Bedrock (placeholder)

### Preprocessing (`utils/rag_preprocessing.py`)

Text cleaning and chunking:

```python
from utils.rag_preprocessing import clean_text, chunk_text, enrich_metadata

cleaned = clean_text("<html>text</html>", remove_html=True)
chunks = chunk_text(cleaned, strategy="fixed_size", chunk_size=500)
enriched = enrich_metadata(chunks[0], source="doc.pdf", page=1)
```

**Chunking Strategies:**
- `fixed_size`: Fixed character count
- `recursive`: Recursive splitting by separators
- `sentence`: Sentence-based splitting

## RAG Pipeline Modules

### Index Builder (`rag/index_builder.py`)

Builds vector indexes from documents:

```python
from rag.index_builder import IndexBuilder

builder = IndexBuilder(
    chunk_size=500,
    embedding_model="text-embedding-ada-002"
)
metadata = builder.build(["doc1.txt", "doc2.pdf"], "./index")
```

**Process:**
1. Load documents
2. Clean and chunk text
3. Generate embeddings
4. Store in vector store

### Retriever (`rag/retriever.py`)

Retrieves relevant documents:

```python
from rag.retriever import Retriever

retriever = Retriever(vector_store=store, top_k=5)
results = retriever.retrieve("What is RAG?")
```

**Features:**
- Vector similarity search
- Metadata filtering
- Configurable top-k

### Re-ranker (`rag/re_ranker.py`)

Re-ranks retrieval results:

```python
from rag.re_ranker import ReRanker

reranker = ReRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank(query, initial_results)
```

**Features:**
- Cross-encoder re-ranking
- Score combination
- Optional (can be skipped)

### Prompt Builder (`rag/prompt_builder.py`)

Builds prompts for LLM:

```python
from rag.prompt_builder import PromptBuilder

builder = PromptBuilder(style="standard", include_sources=True)
prompt = builder.build(query, context_docs)
```

**Styles:**
- `standard`: Full context with instructions
- `concise`: Minimal formatting
- `verbose`: Detailed instructions
- `json`: JSON-structured output

### Inference (`rag/inference.py`)

End-to-end RAG query:

```python
from rag.inference import RAGInference

inference = RAGInference(
    index_path="./index",
    top_k=5,
    llm_model="gpt-4"
)
result = inference.query("What is RAG?")
```

**Returns:**
- `answer`: Generated answer
- `retrieved_docs`: Retrieved documents
- `prompt`: Final prompt
- `metadata`: Additional info

### Evaluation (`rag/evaluation.py`)

Evaluates RAG performance:

```python
from rag.evaluation import RAGEvaluator

evaluator = RAGEvaluator()
metrics = evaluator.evaluate(
    queries=["What is RAG?"],
    ground_truth=["RAG is..."],
    predictions=["RAG stands for..."]
)
```

**Metrics:**
- Exact match rate
- Token overlap F1
- Hit rate
- Recall at k

## Plugging in Different Components

### Using a Different Vector Store

```python
from rag.storage.vector_store import get_vector_store

# Use FAISS instead of Pinecone
store = get_vector_store("faiss", index_path="./index")
```

### Using a Different LLM

```python
from utils.llm_utils import call_llm

# Use Bedrock instead of OpenAI
response = call_llm(
    prompt="...",
    model="anthropic.claude-v2",
    provider="bedrock"
)
```

### Custom Chunking Strategy

```python
from utils.rag_preprocessing import chunk_text

# Use recursive chunking
chunks = chunk_text(text, strategy="recursive", chunk_size=500)
```

## Best Practices

1. **Use Configuration**: Don't hardcode values, use config
2. **Validate Inputs**: Always validate before processing
3. **Log Key Events**: Log important operations for debugging
4. **Handle Errors**: Use try/except with meaningful messages
5. **Test Components**: Write tests for each module
6. **Document Assumptions**: Document any assumptions in code

