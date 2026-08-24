"""Text chunker that extracts Markdown frontmatter as metadata.

Documents starting with a YAML frontmatter block (---) will have it
parsed and returned alongside the chunks. Preserves markdown tables intact.
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
    """Split text into chunks, preserving markdown tables intact."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    metadata, main_text = parse_frontmatter(text)

    # Split main text into structural blocks (paragraphs or tables) by double newlines
    raw_blocks = main_text.split("\n\n")
    blocks = []
    
    # Pre-process blocks: identify tables and group them
    for rb in raw_blocks:
        rb_strip = rb.strip()
        if not rb_strip:
            continue
        # A block is a table if it contains at least one line starting/ending with | and a divider line
        lines = rb_strip.split("\n")
        is_table = False
        if len(lines) >= 2:
            has_pipe = any("|" in line for line in lines)
            has_divider = any(re.match(r'^[\s|:-]+$', line) for line in lines if "---" in line or "-:-" in line)
            if has_pipe and has_divider:
                is_table = True
        
        blocks.append({"text": rb_strip, "is_table": is_table, "length": len(rb_strip)})

    chunks = []
    current_chunk_blocks = []
    current_len = 0

    for block in blocks:
        # If a single block is larger than chunk_size, we just have to yield it or split it
        if block["length"] > size:
            # If it's a table, do NOT split it to preserve structured data
            if block["is_table"]:
                if current_chunk_blocks:
                    chunks.append("\n\n".join(current_chunk_blocks))
                    current_chunk_blocks = []
                    current_len = 0
                chunks.append(block["text"])
                continue
            else:
                # Split normal large paragraphs into sentences
                sentences = re.split(r'(?<=[.!?])\s+', block["text"])
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if current_len + len(sentence) > size and current_chunk_blocks:
                        chunks.append("\n\n".join(current_chunk_blocks))
                        # Keep overlap blocks roughly
                        current_chunk_blocks = current_chunk_blocks[-1:] if current_chunk_blocks else []
                        current_len = sum(len(b) + 2 for b in current_chunk_blocks)
                    current_chunk_blocks.append(sentence)
                    current_len += len(sentence) + 2
                continue

        # Normal fitting blocks
        if current_len + block["length"] > size and current_chunk_blocks:
            chunks.append("\n\n".join(current_chunk_blocks))
            # Hand overlap blocks
            overlap_len = 0
            overlap_blocks = []
            for cb in reversed(current_chunk_blocks):
                if overlap_len + len(cb) < overlap:
                    overlap_blocks.insert(0, cb)
                    overlap_len += len(cb) + 2
                else:
                    break
            current_chunk_blocks = overlap_blocks
            current_len = sum(len(b) + 2 for b in current_chunk_blocks)

        current_chunk_blocks.append(block["text"])
        current_len += block["length"] + 2

    if current_chunk_blocks:
        chunks.append("\n\n".join(current_chunk_blocks))

    return chunks, metadata
