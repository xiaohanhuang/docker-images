"""Document loading and chunking strategies for the text_chunker component."""


def _load_document(path: str) -> str:
    """Load text from a document file."""
    import os

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    elif ext in [".html", ".htm"]:
        from bs4 import BeautifulSoup

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "lxml")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator="\n", strip=True)

    else:
        # Explicitly supported text-like formats
        text_extensions = {
            ".txt",
            ".md",
            ".rst",
            ".log",
            ".json",
            ".csv",
            ".tsv",
            ".yaml",
            ".yml",
        }
        if ext not in text_extensions:
            raise ValueError(
                f"Unsupported file extension for document loading: "
                f"{ext or '<no extension>'}. "
                f"Supported formats: pdf, html/htm, {', '.join(sorted(text_extensions))}"
            )
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def _chunk_text(text: str, strategy: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks based on the specified strategy."""
    if strategy == "character":
        return _chunk_by_characters(text, chunk_size, chunk_overlap)
    elif strategy == "token":
        return _chunk_by_tokens(text, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        return _chunk_by_sentences(text, chunk_size, chunk_overlap)
    elif strategy == "recursive":
        return _chunk_recursive(text, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _chunk_by_characters(text: str, size: int, overlap: int) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + size
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        start += size - overlap

    return chunks


def _chunk_by_tokens(text: str, size: int, overlap: int) -> list[str]:
    """Split text into token-based chunks using tiktoken."""
    import tiktoken

    # Use GPT-4 tokenizer (cl100k_base encoding)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    num_tokens = len(tokens)

    while start < num_tokens:
        end = start + size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text)
        start += size - overlap

    return chunks


def _chunk_by_sentences(text: str, size: int, overlap: int) -> list[str]:
    """Split text by sentences, grouping them to approximate chunk_size."""
    import nltk

    # NLTK data must be pre-baked into the image (see data-cpu Dockerfile)
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK punkt tokenizer data is missing. "
            "Ensure the data-cpu image was built with NLTK data baked in."
        ) from exc

    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_length + sentence_len > size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep last few sentences for overlap
            overlap_chars = 0
            overlap_sentences: list[str] = []
            for sent in reversed(current_chunk):
                if overlap_chars + len(sent) <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_chars += len(sent)
                else:
                    break
            current_chunk = overlap_sentences
            current_length = overlap_chars

        current_chunk.append(sentence)
        current_length += sentence_len

    # Add remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _chunk_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Recursively split text by paragraphs, then sentences, then characters."""
    import re

    import nltk

    # NLTK data must be pre-baked into the image (see data-cpu Dockerfile)
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK punkt tokenizer data is missing. "
            "Ensure the data-cpu image was built with NLTK data baked in."
        ) from exc

    # Level 1: Split by double newlines (paragraphs)
    paragraphs = re.split(r"\n\n+", text)

    chunks = []
    for para in paragraphs:
        if len(para) <= size:
            # Paragraph fits in one chunk
            if para.strip():
                chunks.append(para)
        else:
            # Level 2: Split by sentences
            sentences = nltk.sent_tokenize(para)
            current_chunk: list[str] = []
            current_length = 0

            for sentence in sentences:
                sentence_len = len(sentence)

                if sentence_len > size:
                    # Level 3: Split by characters
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0

                    char_chunks = _chunk_by_characters(sentence, size, overlap)
                    chunks.extend(char_chunks)
                else:
                    if current_length + sentence_len > size and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        # Keep overlap
                        overlap_chars = 0
                        overlap_sentences: list[str] = []
                        for sent in reversed(current_chunk):
                            if overlap_chars + len(sent) <= overlap:
                                overlap_sentences.insert(0, sent)
                                overlap_chars += len(sent)
                            else:
                                break
                        current_chunk = overlap_sentences
                        current_length = overlap_chars

                    current_chunk.append(sentence)
                    current_length += sentence_len

            if current_chunk:
                chunks.append(" ".join(current_chunk))

    return chunks
