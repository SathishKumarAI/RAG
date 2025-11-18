# Refactoring Decisions & Implementation Log

**Date:** 2024  
**Status:** Phase 1 & 2 Complete - File moves done, import updates pending

---

## Executive Summary

This document records all decisions made during the project refactoring, what was changed, and why. The refactoring was done to:
1. Remove duplicate files
2. Organize into a cleaner, more standard structure
3. Consolidate scattered resources (notebooks, configs, deployment files)

---

## Decisions Made

### 1. Duplicate File Removal

#### LICENSE Files
- **Decision:** Keep root `LICENSE` (Apache 2.0, 11KB), delete `rag-pipeline/LICENSE` (empty)
- **Rationale:** Root LICENSE is the canonical version, rag-pipeline/LICENSE was empty
- **Action Taken:** ✅ Deleted `rag-pipeline/LICENSE`

#### README Files
- **Decision:** Merge `rag-pipeline/README.md` content into root `README.md`
- **Rationale:** Single source of truth at root level, better visibility
- **Action Taken:** ✅ Merged content, deleted `rag-pipeline/README.md`

### 2. Directory Structure Reorganization

#### Main Python Package
- **Decision:** Move `rag_pipeline/` → `src/rag_pipeline/`
- **Rationale:** Standard Python project layout (PEP 420), separates source code from other files
- **Action Taken:** ✅ Moved directory
- **Impact:** ⚠️ **BREAKING CHANGE** - All imports need updating (see below)

#### Configuration Directory
- **Decision:** Rename `config/` → `configs/`
- **Rationale:** Plural form is more standard, clearer intent
- **Action Taken:** ✅ Renamed directory
- **Impact:** ⚠️ **BREAKING CHANGE** - All config imports need updating

#### Notebooks Consolidation
- **Decision:** Create `notebooks/` directory with subdirectories:
  - `notebooks/tutorials/` - Tutorial notebooks
  - `notebooks/utilities/` - Utility notebooks
- **Rationale:** Centralized location for all Jupyter notebooks, easier to find and manage
- **Action Taken:** ✅ Created structure and moved:
  - `samples/multimodal_rag_basic.ipynb` → `notebooks/tutorials/`
  - `scripts/dir_create/dir.ipynb` → `notebooks/utilities/dir_create.ipynb`

#### Deployment Configs Consolidation
- **Decision:** Move `gitops/` → `deploy/gitops/`
- **Rationale:** All deployment-related configs in one place (Helm, K8s, GitOps)
- **Action Taken:** ✅ Moved directory
- **Impact:** ⚠️ **BREAKING CHANGE** - CI/CD and deployment scripts may need path updates

### 3. Root-Level File Organization

#### Requirements Files
- **Decision:** Move to root level with descriptive names:
  - `rag-pipeline/requirements.txt` → `requirements.txt` (main)
  - `rag-pipeline/samples/requirements_basic_rag.txt` → `requirements-tutorial.txt`
  - Keep `rag-pipeline/infra/cdk/requirements.txt` in place (CDK-specific)
- **Rationale:** Root-level requirements are standard, tutorial requirements are separate
- **Action Taken:** ✅ Moved files

#### Build/Config Files
- **Decision:** Move to root level:
  - `rag-pipeline/pyproject.toml` → root
  - `rag-pipeline/setup.cfg` → root
  - `rag-pipeline/Makefile` → root
- **Rationale:** These are project-level files, belong at root
- **Action Taken:** ✅ Moved files

### 4. Empty/Unused Directory Cleanup

#### Removed Directories
- **Decision:** Delete empty/unused directories:
  - `samples/example_env/` (empty)
  - `samples/input_pdfs/` (only README, actual data in `data/raw/`)
  - `scripts/dir_create/` (empty after moving notebook)
- **Rationale:** Clean up clutter, reduce confusion
- **Action Taken:** ✅ Deleted directories

---

## Files Changed

### Deleted (4 items)
1. `rag-pipeline/LICENSE` - Empty duplicate
2. `rag-pipeline/README.md` - Merged into root
3. `rag-pipeline/samples/example_env/` - Empty directory
4. `rag-pipeline/samples/input_pdfs/` - Unused (data in data/raw/)

### Moved (10 items)
1. `rag-pipeline/rag_pipeline/` → `rag-pipeline/src/rag_pipeline/`
2. `rag-pipeline/config/` → `rag-pipeline/configs/`
3. `rag-pipeline/gitops/` → `rag-pipeline/deploy/gitops/`
4. `rag-pipeline/samples/multimodal_rag_basic.ipynb` → `rag-pipeline/notebooks/tutorials/multimodal_rag_basic.ipynb`
5. `rag-pipeline/scripts/dir_create/dir.ipynb` → `rag-pipeline/notebooks/utilities/dir_create.ipynb`
6. `rag-pipeline/requirements.txt` → `requirements.txt` (root)
7. `rag-pipeline/samples/requirements_basic_rag.txt` → `requirements-tutorial.txt` (root)
8. `rag-pipeline/pyproject.toml` → `pyproject.toml` (root)
9. `rag-pipeline/setup.cfg` → `setup.cfg` (root)
10. `rag-pipeline/Makefile` → `Makefile` (root)

### Created (3 directories)
1. `rag-pipeline/src/` - For main Python package
2. `rag-pipeline/notebooks/tutorials/` - For tutorial notebooks
3. `rag-pipeline/notebooks/utilities/` - For utility notebooks

---

## Breaking Changes & Required Updates

### ⚠️ Critical: Import Path Updates Needed

#### 1. Python Imports
**Old:**
```python
from rag_pipeline.ingestion import local_loader
from config.settings import Settings
```

**New:**
```python
from src.rag_pipeline.ingestion import local_loader
from configs.settings import Settings
```

**OR** (recommended) - Update `PYTHONPATH` or `sys.path`:
```python
# Add src/ to path, then use:
from rag_pipeline.ingestion import local_loader
from configs.settings import Settings
```

**Files to Update:**
- All Python files in `api/`, `mlops/`, `tests/`
- All scripts in `scripts/`
- Dockerfiles
- Any CI/CD configs

#### 2. Config Imports
**Old:**
```python
from config import settings
import config.config as cfg
```

**New:**
```python
from configs import settings
import configs.config as cfg
```

#### 3. Deployment Scripts
- Update any scripts referencing `gitops/` → `deploy/gitops/`
- Update CI/CD pipelines if they reference old paths
- Update Dockerfile COPY commands if they reference old paths

#### 4. Notebook Paths
- Update any hardcoded paths in notebooks
- Update references to `samples/` if notebooks reference it

---

## Files That Need Manual Review

### High Priority
1. **All test files** (`tests/*.py`) - Update imports
2. **API files** (`api/*.py`) - Update imports
3. **Scripts** (`scripts/*.sh`) - Update paths
4. **Dockerfiles** (`docker/Dockerfile.*`) - Update COPY paths
5. **CI/CD configs** (if any) - Update paths

### Medium Priority
1. **Documentation** - Update any path references
2. **Example code** - Update import examples
3. **README files** - Update structure documentation

### Low Priority
1. **Comments** - Update any path references in comments
2. **Docstrings** - Update any path examples

---

## Verification Checklist

After import updates, verify:
- [ ] All tests pass: `pytest tests/`
- [ ] API starts: `python -m api.fastapi_app` (or equivalent)
- [ ] Scripts work: Test all scripts in `scripts/`
- [ ] Docker builds: `docker build -f docker/Dockerfile.api .`
- [ ] Notebooks run: Test notebooks in `notebooks/`
- [ ] CI/CD passes (if applicable)

---

## Migration Guide for Developers

### For Local Development

1. **Update Python Path:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/rag-pipeline/src"
   ```
   Or add to your shell profile.

2. **Update Imports:**
   - Change `from rag_pipeline` → `from src.rag_pipeline` OR
   - Add `src/` to PYTHONPATH and keep `from rag_pipeline`
   - Change `from config` → `from configs`

3. **Update Scripts:**
   - Check all shell scripts for path references
   - Update any hardcoded paths

### For Docker/Deployment

1. **Update Dockerfiles:**
   - Update COPY commands to new paths
   - Update WORKDIR if needed
   - Update PYTHONPATH in Dockerfile

2. **Update Deployment Configs:**
   - Update any volume mounts
   - Update any path references in K8s/Helm configs

---

## Rollback Plan

If issues arise, the refactoring can be partially rolled back:

1. **Move src/rag_pipeline/ back to rag_pipeline/**
2. **Move configs/ back to config/**
3. **Move deploy/gitops/ back to gitops/**
4. **Move notebooks back to original locations**
5. **Move root files back to rag-pipeline/**

However, **deleted files cannot be recovered** without git history:
- `rag-pipeline/LICENSE` (was empty anyway)
- `rag-pipeline/README.md` (content merged into root)
- Empty directories (can be recreated if needed)

---

## Next Steps

1. ✅ **Phase 1 & 2 Complete:** File moves and deletions done
2. ⏳ **Phase 3 Pending:** Update all import paths
3. ⏳ **Phase 4 Pending:** Update documentation
4. ⏳ **Phase 5 Pending:** Run full test suite
5. ⏳ **Phase 6 Pending:** Update CI/CD if applicable

---

## Notes

- All moves were done using `refactor_moves.py` script for consistency
- Original file structure is preserved in git history
- This document should be updated as import fixes are completed
- Consider adding a `MIGRATION.md` in root for user-facing migration guide

---

## Questions/Issues?

If you encounter issues:
1. Check this document first
2. Review `refactoring_analysis.md` for structure details
3. Check git history for original file locations
4. Update this document with any new findings

