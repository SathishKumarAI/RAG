# Basic Multimodal RAG Pipeline Tutorial

This is a beginner-friendly, step-by-step tutorial for building a multimodal RAG (Retrieval-Augmented Generation) pipeline.

## What This Tutorial Covers

This notebook implements a complete RAG pipeline that:

1. **Parses PDFs** into atomic elements (text, titles, tables, images) using the `unstructured` library
2. **Chunks content** by title, grouping related paragraphs with their tables/images
3. **Handles hybrid chunks** (text + tables/images) with basic summarization
4. **Builds a vector store** using Chroma for semantic search
5. **Demonstrates retrieval** by querying the vector store

## Prerequisites

### System Dependencies

Install these system-level dependencies first:

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils tesseract-ocr libmagic1
```

**macOS:**
```bash
brew install poppler tesseract libmagic
```

**Windows:**
- Install poppler from: https://github.com/oschwartz10612/poppler-windows/releases
- Install tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Install libmagic (may require additional setup)

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements_basic_rag.txt
```

Or install individually:
```bash
pip install unstructured[pdf]
pip install langchain-core langchain-chroma langchain-openai
pip install python-dotenv
```

### API Keys

For embeddings, you'll need an OpenAI API key:

1. Create a `.env` file in the project root
2. Add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

**Alternative:** You can use local embeddings instead (see the notebook for instructions).

## Usage

1. Open the notebook: `multimodal_rag_basic.ipynb`
2. Run all cells in order
3. Update the PDF path in the final cell to point to your PDF file
4. Execute the main pipeline

## Project Structure

```
samples/
├── multimodal_rag_basic.ipynb    # Main tutorial notebook
├── requirements_basic_rag.txt     # Python dependencies
├── README_basic_rag.md            # This file
└── input_pdfs/                    # Place your PDFs here
```

## Features

- ✅ **Beginner-friendly**: Clear comments and step-by-step explanations
- ✅ **Modular functions**: Each task is a separate, testable function
- ✅ **Debug-friendly**: Helper functions to inspect intermediate results
- ✅ **Jupyter-optimized**: Designed for interactive learning
- ✅ **No LLM required**: Basic version uses rule-based summarization

## Next Steps

After completing this tutorial, you can:

- Add an LLM to generate final answers from retrieved chunks
- Experiment with different chunking strategies
- Try different embedding models (local or cloud-based)
- Add more sophisticated summarization for hybrid chunks
- Integrate with the full RAG pipeline in the parent project

## Troubleshooting

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements_basic_rag.txt`
- Check that system dependencies (poppler, tesseract) are installed

### PDF Parsing Issues
- Ensure the PDF path is correct
- Try a simpler PDF first to test the pipeline
- Check that poppler is properly installed

### Embedding Errors
- Verify your `OPENAI_API_KEY` is set in `.env`
- Or switch to local embeddings (see notebook comments)

### Memory Issues
- For large PDFs, consider processing in batches
- Reduce `max_characters` in chunking parameters

## License

Same as the parent RAG_Mini project.

