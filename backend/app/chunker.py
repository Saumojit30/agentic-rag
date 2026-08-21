"""Text chunker that extracts Markdown frontmatter as metadata.

Documents starting with a YAML frontmatter block (---) will have it
parsed and returned alongside the chunks.
"""

import re
import yaml

from .config import settings

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata, remaining_text)."""
    if not text.startswith("---"):
        return {}, text

    # Find the second '---' that closes the frontmatter
    match = re.search(r'---\s*\n(.*?)\n---(.*)', text, re.DOTALL)
    if not match:
        return {}, text

    yaml_content = match.group(1)
    remaining_text = match.group(2).strip()
    
    try:
        metadata = yaml.safe_load(yaml_content)
        if not isinstance(metadata, dict):
            metadata = {}
        return metadata, remaining_text
    except yaml.YAMLError:
        # If parsing fails, fall back to treating the whole document as text
        return {}, text

def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> tuple[list[str], dict]:
    """Split text into chunks, returning (chunks, metadata)."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    metadata, main_text = parse_frontmatter(text)

    # Basic sentence-aware splitting
    sentences = re.split(r'(?<=[.!?])\s+', main_text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_len = len(sentence)
        if current_len + sentence_len > size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # keep overlap (last 1-2 sentences roughly)
            keep_idx = max(0, len(current_chunk) - (overlap // 50) - 1)
            current_chunk = current_chunk[keep_idx:]
            current_len = sum(len(s) + 1 for s in current_chunk)
            
        current_chunk.append(sentence)
        current_len += sentence_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks, metadata
