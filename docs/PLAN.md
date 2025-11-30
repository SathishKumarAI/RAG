# RAG Pipeline Refactoring Plan

**Branch:** `feature/rag-refactor-llm-platform`  
**Date:** 2024  
**Status:** In Progress

---

## Executive Summary

This document outlines the comprehensive refactoring plan to transform the RAG pipeline codebase into a clean, modular, testable, production-grade system with a complete RAG implementation.

---

## 1. Codebase Analysis

### 1.1 Existing Modules and Responsibilities

#### Current Structure (`rag-pipeline/src/rag_pipeline/`)

| Module | Current State | Responsibility | Status |
|--------|---------------|----------------|--------|
| `ingestion/` | Partially implemented | Load documents from S3, local, web, Confluence | ⚠️ Needs completion |
| `parsing/` | Partially implemented | Extract text from PDF, HTML, DOCX | ⚠️ Needs completion |
| `chunking/` | Partially implemented | Split documents into chunks | ⚠️ Needs completion |
| `embedding/` | Partially implemented | Generate embeddings (Bedrock, Pinecone) | ⚠️ Needs completion |
| `storage/` | Partially implemented | Store vectors (Pinecone), metadata (DynamoDB), raw (S3) | ⚠️ Needs completion |
| `retrieval/` | Partially implemented | Retrieve relevant chunks | ⚠️ Needs completion |
| `generation/` | Partially implemented | Generate answers using LLM | ⚠️ Needs completion |
| `workflows/` | Partially implemented | LangGraph orchestration | ⚠️ Needs completion |

#### Supporting Infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| API Layer | `api/` | ✅ FastAPI, Lambda handlers exist |
| Configuration | `configs/` | ⚠️ Basic YAML, needs utility module |
| Tests | `tests/` | ⚠️ Structure exists, needs expansion |
| Documentation | `docs/` | ⚠️ Exists but needs enhancement |
| Notebooks | `notebooks/` | ⚠️ One tutorial exists, needs more |

### 1.2 Missing / Incomplete Pieces

#### Critical Missing Components

1. **Utility Modules** (`/utils` or `/src/utils`)
   - ❌ `config.py` - Centralized configuration management
   - ❌ `logging_utils.py` - Structured logging
   - ❌ `io_utils.py` - File I/O helpers
   - ❌ `validation.py` - Input validation
   - ❌ `llm_utils.py` - LLM wrapper functions
   - ❌ `embeddings.py` - Embedding utilities (separate from embedding module)
   - ❌ `rag_preprocessing.py` - Text cleaning and chunking utilities

2. **RAG Pipeline Modules** (`/rag` or `/src/rag`)
   - ❌ `pipelines.py` - High-level orchestration
   - ❌ `index_builder.py` - Build index from raw docs
   - ❌ `retriever.py` - Vector + keyword retrieval (needs refactoring)
   - ❌ `re_ranker.py` - Re-ranking logic
   - ❌ `prompt_builder.py` - Prompt templates (needs refactoring)
   - ❌ `inference.py` - End-to-end query pipeline
   - ❌ `evaluation.py` - Quality metrics

3. **Notebooks**
   - ❌ `exploration.ipynb` - Data exploration
   - ❌ `rag_pipeline_demo.ipynb` - End-to-end demo
   - ❌ `embeddings_test.ipynb` - Embedding testing
   - ❌ `evaluation_and_debug.ipynb` - Evaluation examples

4. **Documentation**
   - ❌ `docs/overview.md` - System overview
   - ❌ `docs/architecture.md` - Component architecture with diagrams
   - ❌ `docs/implementation_guide.md` - Module-by-module guide
   - ❌ `docs/testing_guide.md` - Testing documentation
   - ❌ `docs/gaps_and_todo.md` - Known gaps and future work

### 1.3 Duplicate Logic and Poor Abstractions

**Issues Identified:**

1. **Configuration Management**
   - Config scattered across YAML files and code
   - No centralized config loader
   - No environment variable management

2. **Logging**
   - No structured logging utility
   - Inconsistent logging patterns

3. **Error Handling**
   - Inconsistent error handling across modules
   - No standardized error types

4. **Embedding Logic**
   - Embedding logic mixed with storage logic
   - No clear separation of concerns

5. **Retrieval Logic**
   - Retrieval mixed with generation
   - No clear interface for different retrieval strategies

### 1.4 Inconsistent Naming and Folder Structure

**Issues:**

1. Module names use `stepN_` prefix (e.g., `step3_chunk_text.py`) - should be descriptive
2. No clear separation between utilities and core logic
3. RAG-specific logic mixed with general utilities

### 1.5 RAG-Relevant Pieces Already Present

**Existing Components:**

✅ **Vector Store:** Pinecone integration exists  
✅ **Metadata Store:** DynamoDB integration exists  
✅ **Raw Storage:** S3 integration exists  
✅ **Embedding Providers:** AWS Bedrock, Pinecone  
✅ **Chunking:** Basic chunking strategies exist  
✅ **Parsing:** PDF, HTML, DOCX extractors exist  
✅ **API Layer:** FastAPI endpoints exist  
✅ **Workflows:** LangGraph workflows exist  

**What Needs Enhancement:**

- Abstract vector store interface (currently tied to Pinecone)
- Standardize chunking strategies
- Add re-ranking capabilities
- Improve evaluation metrics
- Add comprehensive testing

### 1.6 Assumptions About the Domain

**Assumptions Made:**

1. **Document Types:** Primarily PDFs, with support for HTML, DOCX, and text
2. **Vector Store:** Pinecone as primary, but should support alternatives
3. **LLM Provider:** AWS Bedrock, but should support OpenAI and others
4. **Deployment:** AWS-centric (S3, DynamoDB, Lambda, Step Functions)
5. **Scale:** Medium-scale production (not petabyte-scale initially)
6. **Use Cases:** Document Q&A, knowledge base search

---

## 2. Task Breakdown (Priority Order)

### Phase 1: Foundation (Utilities)

**Priority: HIGH**  
**Dependencies: None**

1. ✅ Create `rag-pipeline/src/utils/` directory structure
2. ✅ Implement `config.py` - Configuration management
3. ✅ Implement `logging_utils.py` - Structured logging
4. ✅ Implement `io_utils.py` - File I/O helpers
5. ✅ Implement `validation.py` - Input validation
6. ✅ Implement `llm_utils.py` - LLM wrapper functions
7. ✅ Implement `embeddings.py` - Embedding utilities
8. ✅ Implement `rag_preprocessing.py` - Text cleaning and chunking

**Files to Create:**
- `rag-pipeline/src/utils/__init__.py`
- `rag-pipeline/src/utils/config.py`
- `rag-pipeline/src/utils/logging_utils.py`
- `rag-pipeline/src/utils/io_utils.py`
- `rag-pipeline/src/utils/validation.py`
- `rag-pipeline/src/utils/llm_utils.py`
- `rag-pipeline/src/utils/embeddings.py`
- `rag-pipeline/src/utils/rag_preprocessing.py`

### Phase 2: RAG Pipeline Core

**Priority: HIGH**  
**Dependencies: Phase 1**

1. ✅ Create `rag-pipeline/src/rag/` directory structure
2. ✅ Implement `pipelines.py` - High-level orchestration
3. ✅ Implement `index_builder.py` - Build index from raw docs
4. ✅ Refactor `retriever.py` - Vector + keyword retrieval
5. ✅ Implement `re_ranker.py` - Re-ranking logic
6. ✅ Refactor `prompt_builder.py` - Prompt templates
7. ✅ Implement `inference.py` - End-to-end query pipeline
8. ✅ Implement `evaluation.py` - Quality metrics

**Files to Create/Modify:**
- `rag-pipeline/src/rag/__init__.py`
- `rag-pipeline/src/rag/pipelines.py`
- `rag-pipeline/src/rag/index_builder.py`
- `rag-pipeline/src/rag/retriever.py` (refactor existing)
- `rag-pipeline/src/rag/re_ranker.py`
- `rag-pipeline/src/rag/prompt_builder.py` (refactor existing)
- `rag-pipeline/src/rag/inference.py`
- `rag-pipeline/src/rag/evaluation.py`

### Phase 3: Notebooks

**Priority: MEDIUM**  
**Dependencies: Phase 1, Phase 2**

1. ✅ Create `exploration.ipynb` - Data exploration
2. ✅ Create `rag_pipeline_demo.ipynb` - End-to-end demo
3. ✅ Create `embeddings_test.ipynb` - Embedding testing
4. ✅ Create `evaluation_and_debug.ipynb` - Evaluation examples

**Files to Create:**
- `rag-pipeline/notebooks/exploration.ipynb`
- `rag-pipeline/notebooks/rag_pipeline_demo.ipynb`
- `rag-pipeline/notebooks/embeddings_test.ipynb`
- `rag-pipeline/notebooks/evaluation_and_debug.ipynb`

### Phase 4: Test Suite

**Priority: HIGH**  
**Dependencies: Phase 1, Phase 2**

1. ✅ Enhance `tests/conftest.py` - Test fixtures
2. ✅ Create `tests/test_utils/` - Utility tests
3. ✅ Create `tests/test_rag/` - RAG pipeline tests
4. ✅ Create `tests/test_data/` - Test data directory
5. ✅ Add `pytest.ini` - Test configuration

**Files to Create/Modify:**
- `rag-pipeline/tests/conftest.py` (enhance)
- `rag-pipeline/tests/test_utils/test_config.py`
- `rag-pipeline/tests/test_utils/test_logging_utils.py`
- `rag-pipeline/tests/test_utils/test_io_utils.py`
- `rag-pipeline/tests/test_utils/test_validation.py`
- `rag-pipeline/tests/test_utils/test_llm_utils.py`
- `rag-pipeline/tests/test_utils/test_embeddings.py`
- `rag-pipeline/tests/test_utils/test_rag_preprocessing.py`
- `rag-pipeline/tests/test_rag/test_index_builder.py`
- `rag-pipeline/tests/test_rag/test_retriever.py`
- `rag-pipeline/tests/test_rag/test_prompt_builder.py`
- `rag-pipeline/tests/test_rag/test_inference.py`
- `rag-pipeline/tests/test_data/sample_docs/` (directory)
- `rag-pipeline/tests/test_data/sample_queries.json`
- `rag-pipeline/pytest.ini`

### Phase 5: Documentation

**Priority: MEDIUM**  
**Dependencies: All phases**

1. ✅ Create `docs/overview.md` - System overview
2. ✅ Create `docs/architecture.md` - Component architecture
3. ✅ Create `docs/implementation_guide.md` - Implementation guide
4. ✅ Create `docs/testing_guide.md` - Testing guide
5. ✅ Create `docs/gaps_and_todo.md` - Gaps and future work

**Files to Create:**
- `rag-pipeline/docs/overview.md`
- `rag-pipeline/docs/architecture.md`
- `rag-pipeline/docs/implementation_guide.md`
- `rag-pipeline/docs/testing_guide.md`
- `rag-pipeline/docs/gaps_and_todo.md`

### Phase 6: Final Deliverables

**Priority: HIGH**  
**Dependencies: All phases**

1. ✅ Create `PR_SUMMARY.md` - Pull request summary
2. ✅ Review and finalize all code
3. ✅ Ensure all tests pass
4. ✅ Update README if needed

---

## 3. RAG Roadmap

### Current State
- Basic RAG pipeline exists but is incomplete
- Vector store (Pinecone) integrated
- Basic retrieval and generation implemented
- No evaluation framework
- No re-ranking

### Target State
- ✅ Complete, modular RAG pipeline
- ✅ Abstracted vector store interface
- ✅ Multiple retrieval strategies (vector, keyword, hybrid)
- ✅ Re-ranking support
- ✅ Comprehensive evaluation framework
- ✅ Production-ready error handling and logging
- ✅ Full test coverage
- ✅ Comprehensive documentation

### Future Enhancements (Post-MVP)
- Multi-vector search
- Advanced semantic chunking
- Query expansion and rewriting
- Caching layer
- Async/streaming support
- Multi-tenant support
- Advanced observability (traces, metrics)

---

## 4. Design Decisions

### 4.1 Module Organization

**Decision:** Create separate `/utils` and `/rag` packages

**Rationale:**
- Clear separation of concerns
- Utilities can be reused across the codebase
- RAG modules are focused and testable

### 4.2 Configuration Management

**Decision:** Centralized config module with environment variable support

**Rationale:**
- Single source of truth for configuration
- Easy to override with environment variables
- Type-safe configuration access

### 4.3 Logging

**Decision:** Structured logging with JSON output

**Rationale:**
- Better observability in production
- Easy to parse and analyze logs
- Standard fields for tracing

### 4.4 Vector Store Abstraction

**Decision:** Abstract interface for vector stores

**Rationale:**
- Easy to swap implementations (Pinecone, FAISS, Chroma, etc.)
- Testable with in-memory implementations
- Future-proof for different backends

### 4.5 Testing Strategy

**Decision:** Comprehensive pytest suite with mocks

**Rationale:**
- Fast, deterministic tests
- No external dependencies in unit tests
- Easy to run in CI/CD

---

## 5. Risk Assessment

### Technical Risks

1. **Breaking Changes**
   - **Risk:** Refactoring may break existing integrations
   - **Mitigation:** All changes on feature branch, comprehensive testing

2. **Performance Impact**
   - **Risk:** New abstractions may add overhead
   - **Mitigation:** Profile critical paths, optimize as needed

3. **Complexity**
   - **Risk:** Too many abstractions may make code hard to understand
   - **Mitigation:** Clear documentation, simple interfaces

### Process Risks

1. **Scope Creep**
   - **Risk:** Adding too many features
   - **Mitigation:** Stick to plan, document future enhancements

2. **Testing Coverage**
   - **Risk:** Not enough test coverage
   - **Mitigation:** Aim for >80% coverage, test critical paths

---

## 6. Success Criteria

✅ All utility modules implemented and tested  
✅ Complete RAG pipeline implemented  
✅ All notebooks created and working  
✅ Test suite with >80% coverage  
✅ Comprehensive documentation  
✅ All code follows best practices  
✅ No breaking changes to existing API (if applicable)  
✅ PR_SUMMARY.md created  

---

## 7. Timeline Estimate

- **Phase 1 (Utilities):** 2-3 hours
- **Phase 2 (RAG Core):** 3-4 hours
- **Phase 3 (Notebooks):** 1-2 hours
- **Phase 4 (Tests):** 2-3 hours
- **Phase 5 (Documentation):** 1-2 hours
- **Phase 6 (Final):** 1 hour

**Total:** ~10-15 hours

---

## 8. Next Steps

1. ✅ Create PLAN.md (this document)
2. ⏳ Implement Phase 1 (Utilities)
3. ⏳ Implement Phase 2 (RAG Core)
4. ⏳ Implement Phase 3 (Notebooks)
5. ⏳ Implement Phase 4 (Tests)
6. ⏳ Implement Phase 5 (Documentation)
7. ⏳ Create PR_SUMMARY.md

---

**End of Plan**

