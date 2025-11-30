# Refactoring Summary - Files to Delete/Move

**⚠️ REVIEW THIS BEFORE PROCEEDING ⚠️**

This document lists all files and folders that will be deleted or moved during the refactoring.

---

## Files to DELETE (Duplicates/Empty)

### 1. Duplicate LICENSE
- **DELETE:** `rag-pipeline/LICENSE` (empty file)
- **KEEP:** `LICENSE` (root, Apache 2.0, 11KB)

### 2. Duplicate README (will be merged)
- **DELETE:** `rag-pipeline/README.md` (after merging into root README.md)
- **KEEP:** `README.md` (root, will be enhanced)

### 3. Empty/Unused Directories
- **DELETE:** `rag-pipeline/samples/example_env/` (empty directory)
- **DELETE:** `rag-pipeline/samples/input_pdfs/` (only contains README.md, actual data is in `data/raw/`)
- **DELETE:** `rag-pipeline/scripts/dir_create/` (after moving notebook out)

---

## Files to MOVE (Reorganization)

### Root Level Files (Move from rag-pipeline/ to root)
1. `rag-pipeline/requirements.txt` → `requirements.txt`
2. `rag-pipeline/pyproject.toml` → `pyproject.toml`
3. `rag-pipeline/setup.cfg` → `setup.cfg`
4. `rag-pipeline/Makefile` → `Makefile`
5. `rag-pipeline/samples/requirements_basic_rag.txt` → `requirements-tutorial.txt`

### Directory Reorganization
1. `rag-pipeline/rag_pipeline/` → `src/rag_pipeline/` (main Python package)
2. `rag-pipeline/config/` → `configs/` (rename to plural)
3. `rag-pipeline/gitops/` → `deploy/gitops/` (consolidate deployment configs)

### Notebooks (Consolidate to notebooks/)
1. `rag-pipeline/samples/multimodal_rag_basic.ipynb` → `notebooks/tutorials/multimodal_rag_basic.ipynb`
2. `rag-pipeline/scripts/dir_create/dir.ipynb` → `notebooks/utilities/dir_create.ipynb`

---

## New Directory Structure

### New Directories to Create
- `src/` (for main Python package)
- `notebooks/` (for all Jupyter notebooks)
  - `notebooks/tutorials/`
  - `notebooks/utilities/`
- `configs/` (rename from `config/`)

---

## Files That Stay in Place

These files/folders will NOT be moved:
- `api/` - API code
- `data/` - Data directories (raw/, processed/, chunks/, etc.)
- `deploy/` - Deployment configs (will have gitops/ added to it)
- `docker/` - Docker configs
- `docs/` - Documentation
- `infra/` - Infrastructure as code
- `mlops/` - MLOps code
- `observability/` - Monitoring
- `orchestration/` - Workflows
- `samples/` - Sample data (will be cleaned but stays)
- `scripts/` - Utility scripts (will be cleaned but stays)
- `tests/` - Test files

---

## Impact Summary

### Total Files to Delete: 4 items
- 1 duplicate LICENSE
- 1 README (after merge)
- 2 empty/unused directories

### Total Files to Move: 10 items
- 5 root-level files
- 3 directories (with all contents)
- 2 notebooks

### Estimated Impact
- **Low Risk:** File deletions (duplicates/empty)
- **Medium Risk:** Directory moves (need import path updates)
- **High Risk:** Moving `rag_pipeline/` to `src/rag_pipeline/` (requires all import updates)

---

## Next Steps After Review

1. ✅ Review this summary
2. ✅ Confirm deletions are safe
3. ✅ Confirm moves are acceptable
4. ✅ Proceed with refactoring
5. ✅ Update all import paths
6. ✅ Run tests to verify
7. ✅ Update documentation

---

**Ready to proceed?** The refactoring will be done in phases with verification at each step.

