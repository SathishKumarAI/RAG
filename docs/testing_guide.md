# Testing Guide

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_utils/test_config.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Organization

Tests are organized to match the source structure:

```
tests/
├── conftest.py              # Shared fixtures
├── test_utils/              # Utility tests
│   ├── test_config.py
│   ├── test_validation.py
│   └── ...
├── test_rag/                # RAG pipeline tests
│   ├── test_retriever.py
│   ├── test_prompt_builder.py
│   └── ...
└── test_data/               # Test data
    ├── sample_docs/
    └── sample_queries.json
```

## Writing Tests

### Test Structure

```python
import pytest
from src.utils.validation import validate_document

def test_validate_document_valid():
    """Test validation of valid document."""
    doc = {
        "text": "Valid text",
        "metadata": {}
    }
    validate_document(doc)  # Should not raise

def test_validate_document_invalid():
    """Test validation fails for invalid document."""
    doc = {"metadata": {}}  # Missing text
    with pytest.raises(ValidationError):
        validate_document(doc)
```

### Using Fixtures

```python
# conftest.py
import pytest

@pytest.fixture
def sample_document():
    return {
        "text": "Sample text",
        "metadata": {"source": "test.txt"}
    }

# test_file.py
def test_with_fixture(sample_document):
    validate_document(sample_document)
```

### Mocking External Services

```python
from unittest.mock import Mock, patch

@patch('src.utils.llm_utils.openai')
def test_llm_call(mock_openai):
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    )
    # Test LLM call
```

## Test Data

### Sample Documents

Place test documents in `tests/test_data/sample_docs/`:

```
tests/test_data/sample_docs/
├── doc1.txt
├── doc2.txt
└── ...
```

### Sample Queries

Use `tests/test_data/sample_queries.json` for query test data:

```json
[
    {
        "query": "What is RAG?",
        "expected_answer": "RAG is...",
        "relevant_docs": ["doc1", "doc2"]
    }
]
```

## Test Categories

### Unit Tests

Test individual functions/modules in isolation:

```python
def test_chunk_text_fixed_size():
    text = "A" * 1000
    chunks = chunk_text(text, strategy="fixed_size", chunk_size=200)
    assert len(chunks) > 1
```

### Integration Tests

Test multiple components working together:

```python
def test_end_to_end_rag():
    # Build index
    index = build_index(["doc1.txt"], "./test_index")
    # Query
    result = run_rag_query("What is RAG?", index_path="./test_index")
    assert "answer" in result
```

## Coverage Goals

- **Target Coverage**: >80% for core modules
- **Critical Paths**: 100% coverage for validation, config, core RAG logic
- **Utilities**: >70% coverage acceptable

## Continuous Integration

Tests should run automatically on:
- Pull requests
- Commits to main branch
- Nightly builds

Example CI configuration:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pytest --cov=src --cov-report=xml
```

## Debugging Tests

### Running with Debugger

```bash
pytest --pdb  # Drop into debugger on failure
```

### Verbose Output

```bash
pytest -vv  # Very verbose
pytest -s   # Show print statements
```

### Running Specific Tests

```bash
pytest tests/test_utils/test_config.py::test_config_basic
pytest -k "test_config"  # Run all tests matching pattern
```

## Best Practices

1. **Test Edge Cases**: Empty inputs, None values, extreme values
2. **Test Error Handling**: Verify exceptions are raised correctly
3. **Use Descriptive Names**: Test names should describe what they test
4. **Keep Tests Fast**: Mock slow operations (API calls, file I/O)
5. **Isolate Tests**: Tests should not depend on each other
6. **Clean Up**: Remove temporary files/directories after tests

