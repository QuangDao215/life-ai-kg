"""Document API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.logging import get_logger
from src.db import get_db
from src.db.models import Chunk, Document, Evidence
from src.schemas import (
    ChunkResponse,
    DocumentCreate,
    DocumentResponse,
    DocumentSummary,
    DocumentWithChunks,
    PaginatedResponse,
)

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================


async def get_document_or_404(db: AsyncSession, document_id: UUID) -> Document:
    """Get document by ID or raise 404."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    return document


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new document",
    description="Ingest a new document with metadata. Duplicate PMIDs are rejected.",
)
async def create_document(
    document: DocumentCreate,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Create a new document."""
    # Check for duplicate PMID
    existing = await db.execute(
        select(Document).where(Document.pmid == document.pmid)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document with PMID {document.pmid} already exists",
        )

    # Create document
    db_document = Document(
        pmid=document.pmid,
        title=document.title,
        abstract=document.abstract,
        authors=document.authors,
        journal=document.journal,
        publication_date=document.publication_date,
        doi=document.doi,
        mesh_terms=document.mesh_terms,
        extra_data=document.extra_data,
    )
    db.add(db_document)
    await db.commit()
    await db.refresh(db_document)

    logger.info("Document created", document_id=str(db_document.id), pmid=document.pmid)

    return DocumentResponse.model_validate(db_document)


@router.get(
    "",
    response_model=PaginatedResponse[DocumentSummary],
    summary="List documents",
    description="List all documents with pagination.",
)
async def list_documents(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[DocumentSummary]:
    """List documents with pagination."""
    offset = (page - 1) * page_size

    # Get total count
    count_result = await db.execute(select(func.count(Document.id)))
    total = count_result.scalar() or 0

    # Get documents with chunk counts
    # Subquery for chunk count
    chunk_count_subq = (
        select(Chunk.document_id, func.count(Chunk.id).label("chunk_count"))
        .group_by(Chunk.document_id)
        .subquery()
    )

    # Subquery for processed status (has evidence)
    processed_subq = (
        select(Chunk.document_id)
        .join(Evidence, Evidence.chunk_id == Chunk.id)
        .distinct()
        .subquery()
    )

    # Main query
    query = (
        select(
            Document,
            func.coalesce(chunk_count_subq.c.chunk_count, 0).label("chunk_count"),
            (Document.id.in_(select(processed_subq.c.document_id))).label(
                "has_extractions"
            ),
        )
        .outerjoin(chunk_count_subq, Document.id == chunk_count_subq.c.document_id)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )

    result = await db.execute(query)
    rows = result.all()

    items = [
        DocumentSummary(
            id=row.Document.id,
            pmid=row.Document.pmid,
            title=row.Document.title,
            journal=row.Document.journal,
            publication_date=row.Document.publication_date,
            chunk_count=row.chunk_count,
            has_extractions=row.has_extractions,
            created_at=row.Document.created_at,
        )
        for row in rows
    ]

    return PaginatedResponse.create(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document by ID",
    description="Retrieve a single document with its metadata.",
)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Get a document by ID."""
    document = await get_document_or_404(db, document_id)

    # Get chunk count
    chunk_count_result = await db.execute(
        select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
    )
    chunk_count = chunk_count_result.scalar() or 0

    # Check if processed
    evidence_result = await db.execute(
        select(Evidence.id)
        .join(Chunk, Evidence.chunk_id == Chunk.id)
        .where(Chunk.document_id == document_id)
        .limit(1)
    )
    has_extractions = evidence_result.scalar_one_or_none() is not None

    response = DocumentResponse.model_validate(document)
    response.chunk_count = chunk_count
    response.has_extractions = has_extractions

    return response


@router.get(
    "/{document_id}/chunks",
    response_model=DocumentWithChunks,
    summary="Get document with chunks",
    description="Retrieve a document with all its text chunks.",
)
async def get_document_with_chunks(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentWithChunks:
    """Get a document with its chunks."""
    # Get document with chunks loaded
    result = await db.execute(
        select(Document)
        .options(selectinload(Document.chunks))
        .where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Get evidence chunk IDs
    evidence_result = await db.execute(
        select(Evidence.chunk_id).distinct().where(
            Evidence.chunk_id.in_([c.id for c in document.chunks])
        )
    )
    processed_chunk_ids = {row[0] for row in evidence_result.all()}

    # Build response
    chunks = [
        ChunkResponse(
            id=chunk.id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            token_count=chunk.token_count,
            has_extractions=chunk.id in processed_chunk_ids,
            created_at=chunk.created_at,
        )
        for chunk in sorted(document.chunks, key=lambda c: c.chunk_index)
    ]

    return DocumentWithChunks(
        id=document.id,
        pmid=document.pmid,
        title=document.title,
        abstract=document.abstract,
        authors=document.authors,
        journal=document.journal,
        publication_date=document.publication_date,
        doi=document.doi,
        mesh_terms=document.mesh_terms,
        extra_data=document.extra_data,
        created_at=document.created_at,
        chunk_count=len(chunks),
        has_extractions=len(processed_chunk_ids) > 0,
        chunks=chunks,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Delete a document and all its associated data.",
)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document."""
    document = await get_document_or_404(db, document_id)

    # Delete document (cascades to chunks, evidence)
    await db.delete(document)
    await db.commit()

    logger.info("Document deleted", document_id=str(document_id))
