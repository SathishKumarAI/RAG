docs/implementation_overview.md

This explains  **every major step** ,  **every folder** , and  **what each file does** , written like real enterprise engineering documentation.

---

# **RAG Pipeline – Full Implementation Documentation**

### *AWS + Pinecone + LangChain + LangGraph + Python + GitOps + MLOps*

---

## **📌 Overview**

This project implements a **production-ready Retrieval-Augmented Generation (RAG)** platform using:

* **Python** for ingestion, preprocessing, RAG logic
* **LangChain + LangGraph** for retrieval/generation workflows
* **Pinecone** for vector storage
* **AWS** for compute, storage, metadata, orchestration, and deployment
* **GitOps** via ArgoCD/Flux + Kubernetes
* **MLOps** via MLflow for evaluation, model tracking, and experiments
* **Docker** for packaging workers and API
* **S3** for storing raw PDFs, processed text, MLflow artifacts

This document explains  **every directory** ,  **the steps in the RAG pipeline** , and  **how each file is used** .

---

# 🧩 **1. RAG Pipeline — End-to-End Steps**

A full RAG pipeline follows these phases:

---

## **Step 1 — Connect to Sources (Ingestion)**

Purpose: Identify & fetch documents from different sources.

Location: `rag_pipeline/ingestion/`

Files:

* **step1_connect_sources.py**

  Entry point that orchestrates ingestion from all supported sources.
* **s3_loader.py**

  Loads files from S3 → downloads into `data/raw/`.
* **web_loader.py**

  Fetches web pages, cleans HTML.
* **local_loader.py**

  Loads PDFs, DOCX, and text files from local directories.
* **confluence_loader.py**

  Example enterprise loader for wikis/knowledgebases.

---

## **Step 2 — Extract & Parse Documents**

Purpose: Convert PDFs/HTML/DOCX into clean text.

Location: `rag_pipeline/parsing/`

Files:

* **step2_extract_and_parse.py**

  Coordinates parsing pipeline, writes output → `data/processed/`.
* **pdf_extractor.py**

  Uses `pypdf` or PyMuPDF to get text from PDFs.
* **html_extractor.py**

  Uses BeautifulSoup to get readable text blocks.
* **docx_extractor.py**

  Extracts text using python-docx.
* **text_normalization.py**

  Cleaning rules: whitespace, unicode fixes, lowercasing, metadata stripping.

Output:

`data/processed/document_name.json`

---

## **Step 3 — Chunking Text Into Segments**

Purpose: Break down long text into smaller pieces your LLMs can retrieve efficiently.

Location: `rag_pipeline/chunking/`

Files:

* **step3_chunk_text.py**

  Main chunking workflow; writes to `data/chunks/`.
* **chunk_strategies.py**

  Implements multiple chunking strategies:

  * Recursive character splitter
  * Token-based chunking
  * Semantic clustering
  * Sentence windowing
* **token_counters.py**

  Uses tiktoken to enforce token-level boundaries.

Output:

`data/chunks/{doc_id}/chunk_001.json`

---

## **Step 4 — Embedding & Formatting**

Purpose: Convert small chunks to vector embeddings.

Location: `rag_pipeline/embedding/`

Files:

* **step4_embed_and_format.py**

  Coordinates embedding and pushes to Pinecone.
* **embeddings_pinecone.py**

  Calls OpenAI/Bedrock/HF embeddings → writes vectors into Pinecone index.
* **embeddings_aws_bedrock.py**

  Uses Amazon Titan or Cohere embeddings via AWS.
* **embedding_utils.py**

  Handles batching, error retries, token counting.

Optional local cache:

`data/embeddings_cache/`

---

## **Step 5 — Store vector + metadata**

Purpose: Persist embeddings in Pinecone and metadata in DynamoDB or S3.

Location: `rag_pipeline/storage/`

Files:

* **step5_store_vectorstore.py**

  Main ingestion to Pinecone + metadata store.
* **pinecone_client.py**

  Initializes Pinecone index.
* **metadata_store_dynamodb.py**

  Saves mapping of:

  * doc_id
  * chunk_id
  * original file
  * S3 path
  * embedding metadata
* **s3_raw_store.py**

  Uploads raw PDFs and processed files to S3:

  * `s3://rag-raw-docs/`
  * `s3://rag-processed-docs/`

---

# 🧭 **2. Query → Retrieval → Augmentation → Generation**

These steps power the live RAG responses.

---

## **Retrieval Layer**

Location: `rag_pipeline/retrieval/`

Files:

* **retriever.py**

  Standard LangChain retriever using Pinecone index.
* **retriever_langgraph.py**

  LangGraph node that wraps retrieval logic.

Flow:

1. Take user query
2. Convert to embedding
3. Query Pinecone
4. Return top relevant chunks

---

## **Generation Layer**

Location: `rag_pipeline/generation/`

Files:

* **rag_chain.py**

  LangChain chain for:

  * retrieval
  * relevance filtering
  * template-based answer generation
* **query_to_response.py**

  Defines the public interface for querying the RAG engine.
* **prompts.py**

  Structured system/user prompts.
* **answer_postprocessing.py**

  Cleans hallucinations, extra text, and formats final output.

---

## **Workflows (LangGraph)**

Location: `rag_pipeline/workflows/`

Files:

* **ingest_flow.py**

  Multi-step graph:

  * connect → extract → chunk → embed → store
* **query_flow.py**

  Multi-step graph:

  * preprocess → retrieve → augment → generate → format

---

# ☁️ **3. AWS Integration**

## **Why AWS?**

This project is designed to scale in production using:

* S3 → raw & processed document storage
* DynamoDB → metadata indexing
* Lambda → inference hosting
* Step Functions → ingestion & retraining workflows
* CloudWatch → logs, metrics
* API Gateway → FastAPI routing
* IAM → secure access
* VPC → private networking

---

### **Infrastructure-as-code (IaC)**

Location: `infra/cdk/`

Files:

* **vpc_stack.py**

  Private VPC hosting API/Lambda.
* **rag_lambda_stack.py**

  Deployment for serverless APIs.
* **rag_stepfunctions_stack.py**

  Defines ingestion workflows.
* **rag_api_stack.py**

  Deploys FastAPI behind API Gateway.
* **dynamodb_pinecone_stack.py**

  Metadata tables + Pinecone IAM roles.
* **s3_buckets_stack.py**

  Creates:

  * `rag-raw-docs`
  * `rag-processed-docs`
  * `rag-mlflow-artifacts`

---

# 🧪 **4. Testing Strategy**

Location: `tests/`

Files:

* **test_ingestion.py**

  Validate file discovery & S3/web ingestion.
* **test_parsing.py**

  Validate PDF/HTML/DOCX extraction.
* **test_chunking.py**

  Validate token length, chunk overlap, boundaries.
* **test_embedding.py**

  Mock Pinecone calls; test embedding creation.
* **test_storage.py**

  DynamoDB metadata writing & validation.
* **test_retrieval_generation.py**

  Retrieval correctness from vector store.
* **test_api.py**

  End-to-end FastAPI responses.
* **test_mlflow_integration.py**

  Validate metrics logging.
* **test_langgraph_flows.py**

  Validate entire LangGraph workflows.

---

# 📁 **5. Data Architecture**

Location: `docs/data_architecture.md`

### **Local Dev Storage**

```
data/
 ├── raw/              # PDFs, DOCX, HTML
 ├── processed/        # cleaned extracted text
 ├── chunks/           # small segments
 ├── embeddings_cache/ # optional local vector cache
 └── tmp/
```

### **Production Storage (AWS)**

```
s3://rag-raw-docs/           <-- original PDFs
s3://rag-processed-docs/     <-- cleaned text, json
s3://rag-mlflow-artifacts/   <-- experiments, metrics, evaluation
DynamoDB                     <-- metadata store
Pinecone                     <-- vector embeddings
CloudWatch Logs              <-- observability
```

---

# 🧪 **6. MLOps (MLflow)**

Location: `mlops/`

Files:

* **mlflow_config.py**

  Points MLflow to:

  * local backend (sqlite)
  * or AWS RDS / DynamoDB for tracking
* **mlflow_tracking_server.md**

  How to start an MLflow server with S3 artifact store.
* **pipelines/rag_experiment_pipeline.py**

  Evaluates:

  * chunking strategies
  * embedding models
  * prompts
* **pipelines/eval_pipeline.py**

  Automated evaluation pipeline.
* **metrics/rag_eval_metrics.py**

  RAG quality metrics (recall@k, MRR, nDCG).
* **model_registry/rag_config_registry.py**

  Tracks:

  * chunk sizes
  * embedding models
  * prompt templates
  * retriever configs

MLflow storage:

```
mlruns/
s3://rag-mlflow-artifacts/
```

---

# 🔁 **7. GitOps (ArgoCD / Flux)**

Location: `gitops/`

Files:

* **argocd/app-rag-pipeline.yaml**

  Points ArgoCD → GitHub repo.
* **kustomize/base/**

  Base deployment manifests.
* **kustomize/overlays/dev/**

  Dev environment configs.
* **kustomize/overlays/prod/**

  Production hardened configs.

Purpose:

* GitHub = single source of truth
* Clusters auto-sync to manifests
* Safe rollbacks
* Zero manual deployments

---

# 🐳 **8. Docker + Deployment**

Location: `docker/`

Files:

* **Dockerfile.api**

  FastAPI container → AWS Lambda/ECS/EKS.
* **Dockerfile.worker**

  Background ingest worker.
* **docker-compose.yml**

  Local multi-service setup:

  * API
  * MLflow
  * Pinecone local proxy (optional)
* **nginx.conf**

  Reverse proxy
* **README.md**

  Instructions for building and running containers.

---

# 🌐 **9. API Layer (FastAPI + Lambda)**

Location: `api/`

Files:

* **fastapi_app.py**

  REST API for querying RAG engine.
* **cli_entrypoint.py**

  For command-line querying.
* **lambda_handler.py**

  Wraps RAG engine for AWS Lambda.
* **schemas.py**

  Pydantic models for:

  * request
  * response
  * document metadata

---

# 🧱 **10. Developer Tooling**

* `.pre-commit-config.yaml` — linting, formatting
* `.editorconfig` — consistent editor config
* `.flake8` — Python linting
* `.github/workflows/ci.yml` — tests & build
* `.github/workflows/cd_gitops.yml` — GitOps sync

---

# 🎯 **Conclusion**

This project provides a  **true production-grade RAG system** , supporting:

* Scalable ingestion
* Robust parsing & chunking
* Efficient vector embedding
* Enterprise-grade search
* Multi-layer augmentation
* Secure generation
* CI/CD via GitOps
* Evaluation via MLOps
* Containerized deployment
* AWS-native workflows

Everything is structured so  **you can immediately start building** , iterate fast, and deploy to production without breaking architecture later.
