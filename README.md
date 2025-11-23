# Token Reducer

**Intelligent token reduction library for LLM applications with advanced semantic and structural compression**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/badge/pypi-0.2.0-blue.svg)](https://pypi.org/project/token-reducer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Token Reducer is a Python library designed to reduce token counts in text and code inputs for Large Language Model (LLM) applications while preserving semantic meaning, logical structure, and task-relevant information. Achieve **50-75% token reduction** with advanced semantic compression techniques.

### What's New in v0.2.0

- Semantic Text Compression (68%+ reduction) - Entity abstraction, proposition extraction, hierarchical summarization
- AST-Based Code Compression (73%+ reduction) - Safe Python code minification using Abstract Syntax Trees
- Domain-Specific Handlers (74%+ reduction) - Specialized compression for logs, transcripts, legal documents
- Enhanced Configuration - 8 new advanced parameters for fine-grained control
- Reversibility Mappings - Optional entity/variable restoration
- Progressive Compression - Compress to specific token budgets

### Key Features

- Context-Aware Compression: Task-specific strategies for summarization, RAG, extraction, reasoning, and more
- Multi-Level Compression: Choose between light (5-15%), moderate (20-40%), or aggressive (50-75%) reduction
- Multi-Stage Pipeline: 6-stage intelligent compression (Identification → Segmentation → Redundancy Removal → Semantic Compression → Optimization → Reversibility)
- Text & Code Support: Domain-specific compression for natural language and source code
- Tokenizer Agnostic: Works with OpenAI, Anthropic, HuggingFace, and custom tokenizers
- Fail-Safe Mode: Automatic quality validation with semantic similarity checking
- High Performance: <100ms per 1000 tokens for text, <200ms for code
- Offline Operation: No cloud dependencies, works standalone with intelligent fallbacks

## Installation

### Basic Installation

```bash
pip install token-reducer
```

### With Optional Dependencies

```bash
# For advanced NLP features (entity extraction, NER)
pip install token-reducer[nlp]

# For semantic similarity (embeddings-based deduplication)
pip install token-reducer[similarity]

# For HuggingFace tokenizers
pip install token-reducer[transformers]

# For Anthropic tokenizers
pip install token-reducer[anthropic]

# Install everything
pip install token-reducer[all]
```

## Quick Start

### Basic Text Compression

```python
from token_reducer import compress_text, TaskContext, CompressionLevel

# Compress text for summarization task
result = compress_text(
    text="Your long article text here...",
    task=TaskContext.SUMMARIZATION,
    level=CompressionLevel.MODERATE,
    tokenizer="gpt-4"
)

print(f"Original: {result.original_tokens} tokens")
print(f"Compressed: {result.compressed_tokens} tokens")
print(f"Reduction: {result.reduction_percentage}%")
print(f"\nCompressed text:\n{result.compressed_text}")
```

### Advanced Semantic Compression (NEW in v0.2.0)

```python
from token_reducer import (
    compress_text,
    CompressionConfig,
    CompressionLevel,
    TaskContext
)

# Configure advanced compression
config = CompressionConfig(
    task=TaskContext.SUMMARIZATION,
    level=CompressionLevel.AGGRESSIVE,
    tokenizer="gpt-4",
    # Advanced features
    enable_entity_abstraction=True,  # Replace entities with placeholders
    enable_semantic_dedup=True,      # Remove semantically redundant sentences
    enable_proposition_extraction=True,  # Simplify complex sentences
    semantic_threshold=0.85,         # Similarity threshold for deduplication
    reversible=True,                 # Generate reversibility mappings
    target_tokens=500                # Compress to specific token count
)

result = compress_text("Your text here...", config=config)
```

### AST-Based Code Compression (NEW in v0.2.0)

```python
from token_reducer import PythonASTCompressor

# Create AST compressor
compressor = PythonASTCompressor(
    remove_comments=True,
    remove_docstrings=True,
    rename_variables=True,  # Scope-aware variable renaming
    remove_dead_code=True   # Remove unused imports
)

code = '''
def calculate_sum(first_number, second_number):
    """Calculate the sum of two numbers."""
    # Add the numbers together
    result = first_number + second_number
    return result
'''

compressed_code, rename_map = compressor.compress(code)
print(f"Compressed code:\n{compressed_code}")
# Output: def calculate_sum(a,b):
#     c=a+b
#     return c
```

### Domain-Specific Compression (NEW in v0.2.0)

#### Log File Compression

```python
from token_reducer import LogHandler

handler = LogHandler()

logs = """
[2025-11-23 03:15:01] ERROR: Connection failed
[2025-11-23 03:15:02] ERROR: Connection failed
[2025-11-23 03:15:03] ERROR: Connection failed
[2025-11-23 03:15:04] ERROR: Connection failed
"""

compressed_logs, stats = handler.compress_logs(logs, collapse_threshold=3)
print(compressed_logs)
# Output: [TS] ERROR: Connection failed (repeated 4x)
```

#### Meeting Transcript Compression

```python
from token_reducer import TranscriptHandler

handler = TranscriptHandler()

transcript = """
John Smith: Um, so like, I think we should, you know, move forward.
Jane Doe: Yeah, I mean, that sounds good.
"""

compressed, speaker_map = handler.compress_transcript(transcript)
print(compressed)
# Output: JS: so I think we should, move forward. JD: Yeah, that sounds good.
```

#### Entity Abstraction

```python
from token_reducer import EntityAbstractor

abstractor = EntityAbstractor(use_spacy=False)  # Uses regex fallback

text = "International Business Machines Corporation held a session in Islamabad."
abstracted, entity_map = abstractor.abstract_entities(text, preserve=True)

print(abstracted)
# Output: [ORG1] held a session in [LOC1].
print(entity_map)
# Output: {'[ORG1]': 'International Business Machines Corporation', '[LOC1]': 'Islamabad'}
```

## Performance Benchmarks

| Feature | Token Reduction | Use Case |
|---------|----------------|----------|
| AST Code Compression | 73.4% | Python code minification |
| Log Compression | 74.1% | Server logs, error logs |
| Hierarchical Summarization | 68.4% | Long documents, articles |
| Entity Abstraction | 30.8% | Named entity replacement |
| Proposition Extraction | 18.2% | Sentence simplification |
| Transcript Compression | 27.3% | Meeting notes, conversations |

## Compression Strategies

### Task Types

Token Reducer adapts compression strategies based on your use case:

- SUMMARIZATION: Preserves causal links and chronological order
- RAG: Optimizes for retrieval context (entities, facts, key phrases)
- EXTRACTION: Keeps only fields relevant to extraction target
- REASONING: Preserves premises, key details, and logical connections
- TRANSLATION: Bypasses compression entirely
- CODE_COMPLETION: Preserves function signatures and interfaces
- DEBUGGING: Maintains variable names and error-relevant context
- QUESTION_ANSWERING: Preserves facts and entities for potential questions

### Compression Levels

| Level | Token Reduction | Semantic Similarity | Use Case |
|-------|----------------|---------------------|----------|
| Light | 5-15% | >98% | Maximum safety, minimal loss |
| Moderate | 20-40% | >90% | Balanced compression and quality |
| Aggressive | 50-75% | >80% | Maximum savings, acceptable loss |

## Advanced Features (v0.2.0)

### Content Type Identification

Automatically detects and routes content to appropriate handlers:

```python
from token_reducer import ContentIdentifier

identifier = ContentIdentifier()
content_type = identifier.identify(text)
# Returns: CODE, LOGS, PROSE, TRANSCRIPT, CHAT, LEGAL, or ACADEMIC
```

### Hierarchical Summarization

Multi-level text summarization:

```python
from token_reducer import HierarchicalSummarizer

summarizer = HierarchicalSummarizer()

# Sentence-level
summary = summarizer.summarize(text, level="sentence")

# Paragraph-level
summary = summarizer.summarize(text, level="paragraph", max_sentences=3)

# Document-level
summary = summarizer.summarize(text, level="document", target_ratio=0.5)
```

### Semantic Deduplication

Remove semantically redundant content:

```python
from token_reducer import SemanticDeduplicator

deduplicator = SemanticDeduplicator(
    use_embeddings=False,  # Uses heuristic fallback
    similarity_threshold=0.85
)

unique_sentences = deduplicator.deduplicate(sentences)
```

### Legal Document Compression

Specialized handler for legal text:

```python
from token_reducer import LegalHandler

handler = LegalHandler()
compressed, removed = handler.compress_legal_document(
    document,
    preserve_clauses=True
)

# Extract clauses and definitions
clauses = handler.extract_clauses(document)
definitions = handler.extract_definitions(document)
```

## How It Works

### Multi-Stage Pipeline (v0.2.0)

1. Identification: Detect content type (Code, Logs, Prose, etc.)
2. Segmentation: Break into logical units (sentences, paragraphs, functions)
3. Redundancy Removal: Remove repeated patterns and filler content
4. Semantic Compression: Apply NLP techniques (entity abstraction, summarization)
5. Optimization: Final cleanup and variable shortening
6. Reversibility: Generate mapping for restoration (optional)

### Text Compression Pipeline

1. Normalize: Fix spacing, remove HTML, standardize quotes
2. Prune: Remove duplicates, redundancy, verbose explanations
3. Compress: Extract entities/facts, compact phrasing, reduce adjectives
4. Summarize: Apply task-specific tightening
5. Repack: Shorten sentences, optimize structure

### Code Compression Pipeline

1. Parse AST: Build Abstract Syntax Tree
2. Remove Docstrings: Safe docstring removal
3. Remove Comments: Strip all comments
4. Rename Variables: Scope-aware shortening
5. Dead Code Elimination: Remove unused imports/functions
6. Unparse: Convert back to code

## Use Cases

### Reduce LLM API Costs

```python
# Before: 10,000 tokens × $0.03/1K = $0.30 per request
# After (70% reduction): 3,000 tokens × $0.03/1K = $0.09 per request
# Savings: 70% cost reduction
```

### Fit More Context in Token Limits

```python
from token_reducer import batch_compress_text

# Compress multiple documents to fit in context window
results = batch_compress_text(
    texts=[doc1, doc2, doc3, doc4, doc5],
    task=TaskContext.RAG,
    level=CompressionLevel.MODERATE
)

combined = "\n\n".join(r.compressed_text for r in results)
```

### RAG Pipeline Optimization

```python
# Compress retrieved documents before sending to LLM
retrieved_docs = vector_store.similarity_search(query, k=10)

compressed_docs = [
    compress_text(
        doc.page_content,
        task=TaskContext.RAG,
        level=CompressionLevel.MODERATE
    )
    for doc in retrieved_docs
]

context = "\n\n".join(d.compressed_text for d in compressed_docs)
```

## Configuration Options

### CompressionConfig Parameters

```python
config = CompressionConfig(
    # Basic settings
    task=TaskContext.SUMMARIZATION,
    level=CompressionLevel.MODERATE,
    tokenizer="gpt-4",
    
    # Preservation settings
    preserve_entities=True,
    preserve_numbers=True,
    preserve_facts=True,
    preserve_instructions=True,
    
    # Quality settings
    quality_threshold=0.90,
    enable_fail_safe=True,
    
    # Advanced features (v0.2.0)
    enable_ast_parsing=False,
    enable_semantic_dedup=False,
    enable_entity_abstraction=False,
    enable_proposition_extraction=False,
    use_embeddings=False,
    semantic_threshold=0.85,
    target_tokens=None,
    reversible=False
)
```

## API Reference

### Main Functions

- `compress_text(text, task, level, tokenizer, **kwargs)` - Compress text
- `compress_code(code, task, level, language, tokenizer, **kwargs)` - Compress code
- `batch_compress_text(texts, task, level, tokenizer, **kwargs)` - Batch compression
- `compress_with_config(text, config)` - Compress with configuration object

### Advanced Classes (v0.2.0)

- `PythonASTCompressor` - AST-based code compression
- `EntityAbstractor` - Named entity abstraction
- `PropositionExtractor` - Sentence simplification
- `SemanticDeduplicator` - Semantic deduplication
- `HierarchicalSummarizer` - Multi-level summarization
- `LogHandler` - Log file compression
- `TranscriptHandler` - Transcript compression
- `LegalHandler` - Legal document compression
- `ContentIdentifier` - Content type detection
- `Segmenter` - Content segmentation

## Changelog

### v0.2.0 (2025-11-23)

**Major Features:**
- Added semantic text compression (entity abstraction, proposition extraction, deduplication)
- Added AST-based Python code compression (73%+ reduction)
- Added domain-specific handlers (logs, transcripts, legal documents)
- Added multi-stage pipeline architecture
- Added 8 new configuration parameters
- Added reversibility mappings
- Added progressive compression to token budgets

**Performance:**
- AST Code Compression: 73.4% token reduction
- Log Compression: 74.1% token reduction
- Hierarchical Summarization: 68.4% token reduction

### v0.1.0 (2025-11-22)

- Initial release with core token reduction functionality
- Basic text and code compression
- Task-aware compression strategies
- Multi-level compression support
- Multiple tokenizer support

## Development

### Setup Development Environment

```bash
git clone https://github.com/UsamaTufail31/token-reducer.git
cd token-reducer
pip install -e ".[dev,all]"
pre-commit install
```

### Run Tests

```bash
pytest
pytest --cov=token_reducer --cov-report=html
```

### Code Quality

```bash
black src/ tests/
isort src/ tests/
ruff src/ tests/
mypy src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Token Reducer in your research, please cite:

```bibtex
@software{token_reducer,
  title = {Token Reducer: Intelligent Token Reduction for LLM Applications},
  author = {Tufail, Usama},
  year = {2025},
  version = {0.2.0},
  url = {https://github.com/UsamaTufail31/token-reducer}
}
```

## Support

- PyPI: https://pypi.org/project/token-reducer/
- Repository: https://github.com/UsamaTufail31/token-reducer
- Issues: https://github.com/UsamaTufail31/token-reducer/issues

---

**Created by Usama Tufail** | [GitHub](https://github.com/UsamaTufail31) | [PyPI](https://pypi.org/project/token-reducer/)
