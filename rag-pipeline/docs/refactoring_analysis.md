# Project Refactoring Analysis & Plan

**Date:** 2024  
**Purpose:** Clean project structure, remove duplicates, organize into standard layout

---

## Current Structure Analysis

### Root Level
```
RAG_Mini/
├── LICENSE (Apache 2.0 - 11KB)
├── Notes (4 lines)
├── README.md (minimal - 1 line: "# RAG_Mini")
├── .gitignore
└── rag-pipeline/ (main project)
```

### Main Project (rag-pipeline/)
```
rag-pipeline/
├── LICENSE (empty file - DUPLICATE)
├── README.md (detailed - 7 lines)
├── requirements.txt (main dependencies)
├── pyproject.toml
├── setup.cfg
├── Makefile
├── api/ (FastAPI, Lambda handlers)
├── config/ (YAML configs, settings)
├── data/ (raw/, processed/, chunks/, etc.)
├── deploy/ (Helm, Kubernetes)
├── docker/ (Dockerfiles, compose)
├── docs/ (documentation)
├── gitops/ (ArgoCD, Kustomize) - OVERLAPS with deploy/
├── infra/ (CDK, CloudFormation)
├── mlops/ (MLflow, metrics)
├── mlruns/ (MLflow artifacts - should be in .gitignore)
├── observability/ (CloudWatch, logging)
├── orchestration/ (EventBridge, Step Functions)
├── rag_pipeline/ (main Python package - should be src/)
├── samples/ (tutorials, notebooks, example data)
├── scripts/ (utility scripts + notebook)
└── tests/ (test files)
```

---

## Issues Identified

### 1. Duplicate Files

| File | Location 1 | Location 2 | Decision |
|------|-----------|-------------|----------|
| LICENSE | `LICENSE` (Apache 2.0, 11KB) | `rag-pipeline/LICENSE` (empty) | **Keep root/LICENSE, delete rag-pipeline/LICENSE** |
| README.md | `README.md` (minimal) | `rag-pipeline/README.md` (detailed) | **Merge into root README.md** |
| requirements.txt | `rag-pipeline/requirements.txt` (main) | `rag-pipeline/samples/requirements_basic_rag.txt` (tutorial) | **Keep both, rename tutorial one** |
| requirements.txt | `rag-pipeline/requirements.txt` | `rag-pipeline/infra/cdk/requirements.txt` (CDK-specific) | **Keep both (different purposes)** |

### 2. Scattered Notebooks

| Notebook | Current Location | Issue | Decision |
|----------|-----------------|-------|----------|
| `multimodal_rag_basic.ipynb` | `samples/` | Tutorial notebook mixed with samples | **Move to `notebooks/tutorials/`** |
| `dir.ipynb` | `scripts/dir_create/` | Utility notebook in scripts | **Move to `notebooks/utilities/`** |

### 3. Data Directory Duplication

| Location | Contents | Issue | Decision |
|----------|----------|-------|----------|
| `data/raw/` | `Text Chunking.pdf` | Actual data | **Keep as primary** |
| `samples/input_pdfs/` | `README.md` only | Empty placeholder | **Remove, reference data/raw/** |

### 4. Config Scattering

| Location | Contents | Issue | Decision |
|----------|----------|-------|----------|
| `config/` | YAML configs, settings.py | Main config | **Keep, rename to `configs/`** |
| `samples/example_env/` | Empty directory | Unused | **Remove** |

### 5. Deployment Config Overlap

| Location | Contents | Issue | Decision |
|----------|----------|-------|----------|
| `deploy/` | Helm charts, K8s manifests | Deployment configs | **Merge into single `deploy/`** |
| `gitops/` | ArgoCD, Kustomize | GitOps configs | **Move into `deploy/gitops/`** |

### 6. Structure Issues

| Issue | Current | Proposed |
|-------|---------|----------|
| Main package location | `rag_pipeline/` | **Move to `src/rag_pipeline/`** (standard Python layout) |
| Notebooks location | Scattered | **Consolidate to `notebooks/`** |
| Config location | `config/` | **Rename to `configs/`** (plural, standard) |
| Samples location | `samples/` | **Keep but clean up** |

---

## Proposed New Structure

```
RAG_Mini/
├── LICENSE                          # Single source (Apache 2.0)
├── README.md                        # Merged comprehensive README
├── .gitignore
├── Notes
│
├── src/                             # Main Python package (renamed from rag_pipeline/)
│   └── rag_pipeline/
│       ├── __init__.py
│       ├── ingestion/
│       ├── parsing/
│       ├── chunking/
│       ├── embedding/
│       ├── storage/
│       ├── retrieval/
│       ├── generation/
│       └── workflows/
│
├── notebooks/                       # All Jupyter notebooks
│   ├── tutorials/
│   │   └── multimodal_rag_basic.ipynb
│   └── utilities/
│       └── dir_create.ipynb
│
├── configs/                         # All configuration files (renamed from config/)
│   ├── config.yaml
│   ├── logging.yaml
│   ├── secrets.example.yaml
│   └── settings.py
│
├── data/                            # Data directories (keep as is)
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   ├── embeddings_cache/
│   └── tmp/
│
├── tests/                           # Test files (keep as is)
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py
│
├── docs/                            # Documentation (keep as is)
│   ├── refactoring_analysis.md      # This file
│   ├── refactoring_decisions.md     # Decision log
│   └── ...
│
├── scripts/                         # Utility scripts (cleaned)
│   ├── bootstrap_dev_env.sh
│   ├── deploy_lambda.sh
│   ├── run_local_api.sh
│   ├── run_local_ingest.sh
│   └── sync_data_s3.sh
│
├── deploy/                          # All deployment configs (merged)
│   ├── helm/
│   ├── kubernetes/
│   └── gitops/                      # Moved from gitops/
│       ├── argocd/
│       └── kustomize/
│
├── infra/                           # Infrastructure as code (keep as is)
│   ├── cdk/
│   └── cloudformation/
│
├── docker/                          # Docker configs (keep as is)
│
├── api/                             # API code (keep as is)
│
├── mlops/                           # MLOps (keep as is)
│
├── observability/                   # Monitoring (keep as is)
│
├── orchestration/                   # Workflows (keep as is)
│
├── samples/                         # Sample data and examples (cleaned)
│   ├── queries/
│   └── README_basic_rag.md          # Tutorial README
│
├── requirements.txt                 # Main dependencies (moved to root)
├── requirements-dev.txt             # Development dependencies (new)
├── requirements-tutorial.txt        # Tutorial dependencies (renamed from samples/requirements_basic_rag.txt)
├── pyproject.toml
├── setup.cfg
└── Makefile
```

---

## Detailed Refactoring Plan

### Phase 1: File Consolidation (No Structure Changes)

1. **Delete duplicate LICENSE**
   - Delete: `rag-pipeline/LICENSE` (empty)
   - Keep: `LICENSE` (root, Apache 2.0)

2. **Merge README files**
   - Merge `rag-pipeline/README.md` into root `README.md`
   - Delete: `rag-pipeline/README.md`

3. **Organize requirements files**
   - Keep: `rag-pipeline/requirements.txt` → move to root `requirements.txt`
   - Rename: `rag-pipeline/samples/requirements_basic_rag.txt` → `requirements-tutorial.txt` (root)
   - Keep: `rag-pipeline/infra/cdk/requirements.txt` (CDK-specific, stays in infra/)

4. **Remove empty/unused directories**
   - Delete: `samples/example_env/` (empty)
   - Delete: `samples/input_pdfs/` (only has README, data is in data/raw/)

### Phase 2: Structure Reorganization

5. **Create notebooks/ directory**
   - Move: `samples/multimodal_rag_basic.ipynb` → `notebooks/tutorials/multimodal_rag_basic.ipynb`
   - Move: `scripts/dir_create/dir.ipynb` → `notebooks/utilities/dir_create.ipynb`
   - Delete: `scripts/dir_create/` (empty after move)

6. **Rename config/ to configs/**
   - Rename: `config/` → `configs/`
   - Update all imports/references

7. **Move rag_pipeline/ to src/rag_pipeline/**
   - Create: `src/` directory
   - Move: `rag_pipeline/` → `src/rag_pipeline/`
   - Update all imports/references

8. **Consolidate deployment configs**
   - Move: `gitops/` → `deploy/gitops/`
   - Keep: `deploy/helm/` and `deploy/kubernetes/`

9. **Move requirements to root**
   - Move: `rag-pipeline/requirements.txt` → `requirements.txt` (root)
   - Move: `rag-pipeline/samples/requirements_basic_rag.txt` → `requirements-tutorial.txt` (root)

10. **Move other root-level files**
    - Move: `rag-pipeline/pyproject.toml` → root
    - Move: `rag-pipeline/setup.cfg` → root
    - Move: `rag-pipeline/Makefile` → root

### Phase 3: Update References

11. **Update import paths**
    - Update all `from rag_pipeline` → `from src.rag_pipeline` or adjust PYTHONPATH
    - Update config imports: `from config` → `from configs`
    - Update test imports

12. **Update documentation**
    - Update all docs that reference old paths
    - Update README with new structure

13. **Update scripts**
    - Update paths in shell scripts
    - Update Dockerfile paths
    - Update CI/CD configs if any

---

## Files/Folders to Delete

### Safe to Delete (Duplicates/Empty)
- ✅ `rag-pipeline/LICENSE` (empty duplicate)
- ✅ `rag-pipeline/README.md` (merged into root)
- ✅ `samples/example_env/` (empty directory)
- ✅ `samples/input_pdfs/` (only README, data is in data/raw/)
- ✅ `scripts/dir_create/` (after moving notebook)

### Files to Move (Not Delete)
- 📦 `rag-pipeline/requirements.txt` → `requirements.txt` (root)
- 📦 `rag-pipeline/samples/requirements_basic_rag.txt` → `requirements-tutorial.txt` (root)
- 📦 `rag-pipeline/pyproject.toml` → root
- 📦 `rag-pipeline/setup.cfg` → root
- 📦 `rag-pipeline/Makefile` → root
- 📦 `rag-pipeline/rag_pipeline/` → `src/rag_pipeline/`
- 📦 `rag-pipeline/config/` → `configs/`
- 📦 `rag-pipeline/gitops/` → `deploy/gitops/`
- 📦 `samples/multimodal_rag_basic.ipynb` → `notebooks/tutorials/`
- 📦 `scripts/dir_create/dir.ipynb` → `notebooks/utilities/`

---

## Impact Assessment

### Breaking Changes
1. **Import paths** - All `from rag_pipeline` imports need updating
2. **Config paths** - All `from config` imports need updating
3. **Script paths** - Shell scripts may need path updates
4. **Docker paths** - Dockerfiles may need path updates
5. **CI/CD** - Any CI/CD configs referencing old paths

### Non-Breaking
- Data files stay in same relative locations
- Test structure unchanged
- Documentation structure unchanged

---

## Migration Checklist

- [ ] Backup current repository
- [ ] Create feature branch: `refactor/project-structure`
- [ ] Phase 1: Delete duplicates
- [ ] Phase 2: Reorganize structure
- [ ] Phase 3: Update all references
- [ ] Run tests to verify
- [ ] Update documentation
- [ ] Create migration guide
- [ ] Merge to main

---

## Notes

- `mlruns/` should be in `.gitignore` (MLflow artifacts)
- Consider adding `__pycache__/`, `*.pyc` to `.gitignore` if not already
- Consider adding `*.egg-info/`, `dist/`, `build/` to `.gitignore`

