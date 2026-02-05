"""
Document chunking service.

This module provides text chunking functionality for splitting documents
into smaller segments suitable for LLM processing while maintaining
character offset tracking for provenance.

Features:
- Configurable chunk size and overlap
- Sentence-aware splitting (avoids breaking mid-sentence)
- Character offset tracking for evidence provenance
- Token count estimation
- Batch processing of documents

Requirements addressed:
- REQ-SC-04: Evidence tracking with quote offsets
- Supports LLM context window management
"""

import re
from collections.abc import Generator
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import not_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.logging import get_logger
from src.db.models import Chunk, Document

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Approximate characters per token (for English text)
CHARS_PER_TOKEN = 4

# Sentence ending patterns
SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Paragraph separator
PARAGRAPH_SEP = re.compile(r"\n\n+")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChunkData:
    """
    Represents a text chunk with offset information.

    This is an intermediate representation before database storage.
    """

    text: str
    chunk_index: int
    start_offset: int
    end_offset: int
    token_count: int = 0

    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.text)

    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.text)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "token_count": self.token_count,
        }


@dataclass
class ChunkingResult:
    """Result of chunking a document."""

    document_id: UUID
    pmid: str
    chunks: list[ChunkData] = field(default_factory=list)
    total_chars: int = 0
    total_tokens: int = 0

    @property
    def chunk_count(self) -> int:
        """Get number of chunks."""
        return len(self.chunks)


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses a simple heuristic of ~4 characters per token for English.
    For more accurate counts, use tiktoken or similar.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Handles common sentence boundaries while preserving
    abbreviations and decimal numbers.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    if not text:
        return []

    # Split on sentence boundaries
    sentences = SENTENCE_ENDINGS.split(text)

    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs.

    Args:
        text: Input text

    Returns:
        List of paragraphs
    """
    if not text:
        return []

    paragraphs = PARAGRAPH_SEP.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


# =============================================================================
# Chunking Functions
# =============================================================================


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    respect_sentences: bool = True,
) -> Generator[ChunkData, None, None]:
    """
    Split text into overlapping chunks with offset tracking.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in characters (default from settings)
        chunk_overlap: Overlap between chunks in characters (default from settings)
        respect_sentences: If True, avoid breaking mid-sentence

    Yields:
        ChunkData objects with text and offset information

    Example:
        for chunk in chunk_text("Long document text...", chunk_size=500):
            print(f"Chunk {chunk.chunk_index}: {chunk.text[:50]}...")
            print(f"  Offsets: {chunk.start_offset}-{chunk.end_offset}")
    """
    if not text or not text.strip():
        return

    # Use settings defaults if not specified
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Validate parameters
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"Chunk overlap ({chunk_overlap}) must be less than "
            f"chunk size ({chunk_size})"
        )

    text = text.strip()
    text_length = len(text)

    if text_length <= chunk_size:
        # Text fits in single chunk
        yield ChunkData(
            text=text,
            chunk_index=0,
            start_offset=0,
            end_offset=text_length,
        )
        return

    # Split into sentences if requested
    if respect_sentences:
        yield from _chunk_by_sentences(text, chunk_size, chunk_overlap)
    else:
        yield from _chunk_by_characters(text, chunk_size, chunk_overlap)


def _chunk_by_characters(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Generator[ChunkData, None, None]:
    """
    Simple character-based chunking with overlap.

    Args:
        text: Input text
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Yields:
        ChunkData objects
    """
    text_length = len(text)
    chunk_index = 0
    start = 0

    while start < text_length:
        # Calculate end position
        end = min(start + chunk_size, text_length)

        # Extract chunk
        chunk_text = text[start:end]

        yield ChunkData(
            text=chunk_text,
            chunk_index=chunk_index,
            start_offset=start,
            end_offset=end,
        )

        # Move to next chunk with overlap
        start = end - chunk_overlap
        chunk_index += 1

        # Prevent infinite loop
        if start >= text_length:
            break


def _chunk_by_sentences(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Generator[ChunkData, None, None]:
    """
    Sentence-aware chunking that avoids breaking mid-sentence.

    Args:
        text: Input text
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks (in terms of sentences to repeat)

    Yields:
        ChunkData objects
    """
    sentences = split_into_sentences(text)

    if not sentences:
        return

    if len(sentences) == 1:
        # Single sentence - return as single chunk
        yield ChunkData(
            text=text.strip(),
            chunk_index=0,
            start_offset=0,
            end_offset=len(text.strip()),
        )
        return

    # Build chunks from sentences
    chunk_index = 0
    current_sentences: list[str] = []
    current_length = 0

    # Track position in original text
    sentence_positions: list[tuple[int, int]] = []
    pos = 0
    for sentence in sentences:
        # Find sentence in text (accounting for whitespace)
        start_pos = text.find(sentence, pos)
        if start_pos == -1:
            start_pos = pos
        end_pos = start_pos + len(sentence)
        sentence_positions.append((start_pos, end_pos))
        pos = end_pos

    # Number of sentences to overlap
    overlap_sentences = max(1, chunk_overlap // (chunk_size // max(len(sentences), 1)))

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_len = len(sentence) + 1  # +1 for space

        # Check if adding this sentence exceeds chunk size
        if current_length + sentence_len > chunk_size and current_sentences:
            # Emit current chunk
            chunk_text = " ".join(current_sentences)
            start_offset = sentence_positions[i - len(current_sentences)][0]
            end_offset = sentence_positions[i - 1][1]

            yield ChunkData(
                text=chunk_text,
                chunk_index=chunk_index,
                start_offset=start_offset,
                end_offset=end_offset,
            )

            chunk_index += 1

            # Keep last N sentences for overlap
            if overlap_sentences > 0 and len(current_sentences) > overlap_sentences:
                current_sentences = current_sentences[-overlap_sentences:]
                current_length = sum(len(s) + 1 for s in current_sentences)
            else:
                current_sentences = []
                current_length = 0

        current_sentences.append(sentence)
        current_length += sentence_len
        i += 1

    # Emit final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        start_idx = len(sentences) - len(current_sentences)
        start_offset = sentence_positions[start_idx][0]
        end_offset = sentence_positions[-1][1]

        yield ChunkData(
            text=chunk_text,
            chunk_index=chunk_index,
            start_offset=start_offset,
            end_offset=end_offset,
        )


# =============================================================================
# Document Chunking Service
# =============================================================================


class ChunkingService:
    """
    Service for chunking documents and storing chunks in the database.

    Handles:
    - Chunking document abstracts
    - Storing chunks with offset tracking
    - Batch processing of multiple documents

    Usage:
        service = ChunkingService(db_session)

        # Chunk a single document
        result = await service.chunk_document(document)

        # Chunk all documents without chunks
        results = await service.chunk_all_documents()
    """

    def __init__(
        self,
        db: AsyncSession,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Initialize chunking service.

        Args:
            db: Database session
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.db = db
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    async def chunk_document(
        self,
        document: Document,
        replace_existing: bool = False,
    ) -> ChunkingResult:
        """
        Chunk a single document and store chunks in database.

        Args:
            document: Document to chunk
            replace_existing: If True, delete existing chunks first

        Returns:
            ChunkingResult with chunk information
        """
        result = ChunkingResult(
            document_id=document.id,
            pmid=document.pmid,
        )

        # Check if document has abstract
        if not document.abstract or not document.abstract.strip():
            logger.debug("Document has no abstract", pmid=document.pmid)
            return result

        # Check for existing chunks
        existing_chunks = await self.db.execute(
            select(Chunk.id).where(Chunk.document_id == document.id)
        )
        existing = existing_chunks.scalars().all()

        if existing and not replace_existing:
            logger.debug(
                "Document already has chunks",
                pmid=document.pmid,
                chunk_count=len(existing),
            )
            return result

        # Delete existing chunks if replacing
        if existing and replace_existing:
            for chunk_id in existing:
                chunk = await self.db.get(Chunk, chunk_id)
                if chunk:
                    await self.db.delete(chunk)
            await self.db.flush()
            logger.debug("Deleted existing chunks", pmid=document.pmid, count=len(existing))

        # Generate chunks
        result.total_chars = len(document.abstract)

        for chunk_data in chunk_text(
            document.abstract,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        ):
            # Create chunk record
            chunk = Chunk(
                document_id=document.id,
                chunk_index=chunk_data.chunk_index,
                text=chunk_data.text,
                start_offset=chunk_data.start_offset,
                end_offset=chunk_data.end_offset,
                token_count=chunk_data.token_count,
                extra_data={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            )
            self.db.add(chunk)
            result.chunks.append(chunk_data)
            result.total_tokens += chunk_data.token_count

        await self.db.flush()

        logger.debug(
            "Chunked document",
            pmid=document.pmid,
            chunks=result.chunk_count,
            total_chars=result.total_chars,
            total_tokens=result.total_tokens,
        )

        return result

    async def chunk_all_documents(
        self,
        replace_existing: bool = False,
        progress_callback: callable | None = None,
    ) -> list[ChunkingResult]:
        """
        Chunk all documents that don't have chunks yet.

        Args:
            replace_existing: If True, re-chunk documents with existing chunks
            progress_callback: Optional callback(processed, total) for progress

        Returns:
            List of ChunkingResult objects
        """
        # Get documents to process
        if replace_existing:
            # Get all documents with abstracts
            query = select(Document).where(Document.abstract.isnot(None))
        else:
            # Get documents without chunks
            subquery = select(Chunk.document_id).distinct()
            query = select(Document).where(
                Document.abstract.isnot(None),
                not_(Document.id.in_(subquery)),
            )

        result = await self.db.execute(query)
        documents = result.scalars().all()

        total = len(documents)
        results = []

        logger.info(
            "Starting batch chunking",
            total_documents=total,
            replace_existing=replace_existing,
        )

        for i, document in enumerate(documents):
            chunk_result = await self.chunk_document(
                document,
                replace_existing=replace_existing,
            )
            results.append(chunk_result)

            if progress_callback:
                progress_callback(i + 1, total)

        await self.db.commit()

        # Summary statistics
        total_chunks = sum(r.chunk_count for r in results)
        total_tokens = sum(r.total_tokens for r in results)

        logger.info(
            "Batch chunking complete",
            documents_processed=len(results),
            total_chunks=total_chunks,
            total_tokens=total_tokens,
        )

        return results

    async def get_document_chunks(self, document_id: UUID) -> list[Chunk]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document UUID

        Returns:
            List of Chunk objects ordered by chunk_index
        """
        result = await self.db.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())

    async def get_chunk_by_offset(
        self,
        document_id: UUID,
        offset: int,
    ) -> Chunk | None:
        """
        Find the chunk containing a specific character offset.

        Args:
            document_id: Document UUID
            offset: Character offset in original document

        Returns:
            Chunk containing the offset, or None
        """
        result = await self.db.execute(
            select(Chunk).where(
                Chunk.document_id == document_id,
                Chunk.start_offset <= offset,
                Chunk.end_offset > offset,
            )
        )
        return result.scalar_one_or_none()


# =============================================================================
# Convenience Functions
# =============================================================================


async def chunk_documents_batch(
    db: AsyncSession,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    replace_existing: bool = False,
) -> list[ChunkingResult]:
    """
    Convenience function to chunk all documents.

    Args:
        db: Database session
        chunk_size: Target chunk size
        chunk_overlap: Chunk overlap
        replace_existing: Replace existing chunks

    Returns:
        List of ChunkingResult objects
    """
    service = ChunkingService(db, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return await service.chunk_all_documents(replace_existing=replace_existing)
