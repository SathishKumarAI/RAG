# RAG Pipeline with AWS, Pinecone, LangChain & LangGraph

Production-oriented RAG system with:
- Data: local `data/` + S3 buckets for raw PDFs, processed text, and MLflow artifacts.
- Vector store: Pinecone for embeddings; DynamoDB/S3 for metadata.
- MLOps: MLflow (`mlruns/`), evaluation pipelines, and registry-like configs under `mlops/`.
- GitOps: Kubernetes/Helm manifests in `gitops/` and `deploy/`, driven by GitHub Actions.

## Project Structure

See `docs/refactoring_analysis.md` for detailed project structure documentation.