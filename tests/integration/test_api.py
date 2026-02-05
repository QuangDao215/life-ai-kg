"""Integration tests for the LIFE AI Knowledge Graph API.

These tests verify the API endpoints work correctly with the actual database.
Requires a running PostgreSQL database with existing data.
"""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


# =============================================================================
# Health & Root Tests
# =============================================================================


class TestHealthEndpoints:
    """Tests for health and root endpoints."""

    async def test_health_check(self, async_client: AsyncClient) -> None:
        """Test health endpoint returns healthy status."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    async def test_root_endpoint(self, async_client: AsyncClient) -> None:
        """Test root endpoint returns API info."""
        response = await async_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "LIFE AI" in data["service"]
        assert "docs" in data

    async def test_openapi_schema(self, async_client: AsyncClient) -> None:
        """Test OpenAPI schema is generated."""
        response = await async_client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert len(data["paths"]) > 10  # Should have many endpoints


# =============================================================================
# Documents API Tests
# =============================================================================


class TestDocumentsAPI:
    """Tests for documents API endpoints."""

    async def test_list_documents(self, async_client: AsyncClient) -> None:
        """Test listing documents with pagination."""
        response = await async_client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()

        # Check pagination structure
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "pages" in data

        # Verify we have documents (from Phase 2)
        assert data["total"] > 0
        assert len(data["items"]) > 0

        # Check document structure
        doc = data["items"][0]
        assert "id" in doc
        assert "pmid" in doc
        assert "title" in doc
        assert "chunk_count" in doc

    async def test_list_documents_pagination(self, async_client: AsyncClient) -> None:
        """Test document pagination parameters."""
        response = await async_client.get(
            "/api/v1/documents",
            params={"page": 1, "page_size": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 5
        assert data["page"] == 1
        assert data["page_size"] == 5

    async def test_get_document_by_id(self, async_client: AsyncClient) -> None:
        """Test getting a single document by ID."""
        # First get a document ID
        list_response = await async_client.get("/api/v1/documents", params={"page_size": 1})
        doc_id = list_response.json()["items"][0]["id"]

        # Get the document
        response = await async_client.get(f"/api/v1/documents/{doc_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == doc_id
        assert "title" in data
        assert "abstract" in data
        assert "chunk_count" in data

    async def test_get_document_with_chunks(self, async_client: AsyncClient) -> None:
        """Test getting a document with its chunks."""
        # First get a document ID
        list_response = await async_client.get("/api/v1/documents", params={"page_size": 1})
        doc_id = list_response.json()["items"][0]["id"]

        # Get document with chunks
        response = await async_client.get(f"/api/v1/documents/{doc_id}/chunks")

        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert len(data["chunks"]) > 0

        # Check chunk structure
        chunk = data["chunks"][0]
        assert "id" in chunk
        assert "text" in chunk
        assert "chunk_index" in chunk

    async def test_get_nonexistent_document(self, async_client: AsyncClient) -> None:
        """Test 404 for nonexistent document."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await async_client.get(f"/api/v1/documents/{fake_id}")

        assert response.status_code == 404


# =============================================================================
# Entities API Tests
# =============================================================================


class TestEntitiesAPI:
    """Tests for entities API endpoints."""

    async def test_search_entities(self, async_client: AsyncClient) -> None:
        """Test entity search."""
        response = await async_client.get(
            "/api/v1/entities/search",
            params={"query": "parkinson"}
        )

        assert response.status_code == 200
        data = response.json()

        # Check pagination structure
        assert "items" in data
        assert "total" in data

        # Should find Parkinson-related entities
        assert data["total"] > 0

        # Check entity structure
        entity = data["items"][0]
        assert "id" in entity
        assert "name" in entity
        assert "type" in entity

    async def test_search_entities_with_type_filter(self, async_client: AsyncClient) -> None:
        """Test entity search with type filter."""
        response = await async_client.get(
            "/api/v1/entities/search",
            params={"query": "a", "entity_type": "Disease"}
        )

        assert response.status_code == 200
        data = response.json()

        # All results should be Disease type
        for entity in data["items"]:
            assert entity["type"] == "Disease"

    async def test_search_entities_empty_query(self, async_client: AsyncClient) -> None:
        """Test entity search with empty query returns error."""
        response = await async_client.get(
            "/api/v1/entities/search",
            params={"query": ""}
        )

        # Should fail validation
        assert response.status_code == 422

    async def test_get_entity_types(self, async_client: AsyncClient) -> None:
        """Test getting entity type counts."""
        response = await async_client.get("/api/v1/entities/types")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) > 0

        # Check structure
        for item in data:
            assert "type" in item
            assert "count" in item
            assert item["count"] > 0

    async def test_get_entity_by_id(self, async_client: AsyncClient) -> None:
        """Test getting entity by ID."""
        # First search for an entity
        search_response = await async_client.get(
            "/api/v1/entities/search",
            params={"query": "a", "page_size": 1}
        )
        entity_id = search_response.json()["items"][0]["id"]

        # Get the entity
        response = await async_client.get(f"/api/v1/entities/{entity_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == entity_id
        assert "name" in data
        assert "type" in data
        assert "normalized_name" in data

    async def test_get_entity_neighborhood(self, async_client: AsyncClient) -> None:
        """Test getting entity neighborhood."""
        # First search for an entity
        search_response = await async_client.get(
            "/api/v1/entities/search",
            params={"query": "parkinson", "page_size": 1}
        )

        if search_response.json()["total"] == 0:
            pytest.skip("No Parkinson entities found")

        entity_id = search_response.json()["items"][0]["id"]

        # Get neighborhood
        response = await async_client.get(f"/api/v1/entities/{entity_id}/neighborhood")

        assert response.status_code == 200
        data = response.json()

        assert "entity" in data
        assert "relations" in data
        assert "total_relations" in data
        assert data["entity"]["id"] == entity_id

    async def test_get_entity_relations(self, async_client: AsyncClient) -> None:
        """Test getting entity relations with evidence."""
        # First search for an entity with relations
        search_response = await async_client.get(
            "/api/v1/entities/search",
            params={"query": "parkinson", "page_size": 1}
        )

        if search_response.json()["total"] == 0:
            pytest.skip("No Parkinson entities found")

        entity_id = search_response.json()["items"][0]["id"]

        # Get relations
        response = await async_client.get(f"/api/v1/entities/{entity_id}/relations")

        assert response.status_code == 200
        data = response.json()

        assert "items" in data
        assert "total" in data


# =============================================================================
# Graph API Tests
# =============================================================================


class TestGraphAPI:
    """Tests for graph API endpoints."""

    async def test_get_graph_stats(self, async_client: AsyncClient) -> None:
        """Test getting graph statistics."""
        response = await async_client.get("/api/v1/graph/stats")

        assert response.status_code == 200
        data = response.json()

        assert "entity_count" in data
        assert "relation_count" in data
        assert "evidence_count" in data
        assert "entities_by_type" in data
        assert "relations_by_label" in data

        # Should have data from Phase 3
        assert data["entity_count"] > 0
        assert data["relation_count"] > 0

    async def test_get_subgraph(self, async_client: AsyncClient) -> None:
        """Test getting a subgraph from an entity."""
        # First get an entity
        search_response = await async_client.get(
            "/api/v1/entities/search",
            params={"query": "a", "page_size": 1}
        )
        entity_id = search_response.json()["items"][0]["id"]

        # Get subgraph
        response = await async_client.get(
            "/api/v1/graph/subgraph",
            params={
                "entity_id": entity_id,
                "depth": 1,
                "max_nodes": 20
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "nodes" in data
        assert "edges" in data
        assert "center_node" in data
        assert "stats" in data

        # Center node should match request
        assert data["center_node"] == entity_id

        # Should have at least the center node
        assert len(data["nodes"]) >= 1

    async def test_get_subgraph_with_filters(self, async_client: AsyncClient) -> None:
        """Test subgraph with relation type filter."""
        # First get an entity
        search_response = await async_client.get(
            "/api/v1/entities/search",
            params={"query": "a", "page_size": 1}
        )
        entity_id = search_response.json()["items"][0]["id"]

        # Get subgraph with filter
        response = await async_client.get(
            "/api/v1/graph/subgraph",
            params={
                "entity_id": entity_id,
                "depth": 1,
                "relation_types": "ASSOCIATED_WITH",
                "min_confidence": 0.5
            }
        )

        assert response.status_code == 200

    async def test_find_path(self, async_client: AsyncClient) -> None:
        """Test finding paths between entities."""
        # Get two entities
        search_response = await async_client.get(
            "/api/v1/entities/search",
            params={"query": "a", "page_size": 2}
        )

        items = search_response.json()["items"]
        if len(items) < 2:
            pytest.skip("Need at least 2 entities for path finding")

        source_id = items[0]["id"]
        target_id = items[1]["id"]

        # Find path
        response = await async_client.get(
            "/api/v1/graph/path",
            params={
                "source_id": source_id,
                "target_id": target_id,
                "max_hops": 3
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "paths" in data
        assert "source" in data
        assert "target" in data
        assert "path_count" in data
        assert "searched_depth" in data

    async def test_find_path_same_entity(self, async_client: AsyncClient) -> None:
        """Test path finding with same source and target returns error."""
        search_response = await async_client.get(
            "/api/v1/entities/search",
            params={"query": "a", "page_size": 1}
        )
        entity_id = search_response.json()["items"][0]["id"]

        response = await async_client.get(
            "/api/v1/graph/path",
            params={
                "source_id": entity_id,
                "target_id": entity_id,
                "max_hops": 3
            }
        )

        assert response.status_code == 400


# =============================================================================
# Jobs API Tests
# =============================================================================


class TestJobsAPI:
    """Tests for jobs API endpoints."""

    async def test_list_jobs(self, async_client: AsyncClient) -> None:
        """Test listing jobs."""
        response = await async_client.get("/api/v1/jobs")

        assert response.status_code == 200
        data = response.json()

        assert "items" in data
        assert "total" in data

    async def test_get_extraction_progress(self, async_client: AsyncClient) -> None:
        """Test getting extraction progress."""
        response = await async_client.get("/api/v1/jobs/progress/current")

        assert response.status_code == 200
        data = response.json()

        assert "chunks_total" in data
        assert "chunks_processed" in data
        assert "chunks_remaining" in data
        assert "entities_extracted" in data
        assert "relations_extracted" in data

        # Should have data from Phase 2 & 3
        assert data["chunks_total"] > 0

    async def test_get_nonexistent_job(self, async_client: AsyncClient) -> None:
        """Test 404 for nonexistent job."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await async_client.get(f"/api/v1/jobs/{fake_id}")

        assert response.status_code == 404


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for API error handling."""

    async def test_invalid_uuid(self, async_client: AsyncClient) -> None:
        """Test invalid UUID returns 422."""
        response = await async_client.get("/api/v1/documents/invalid-uuid")

        assert response.status_code == 422

    async def test_invalid_pagination(self, async_client: AsyncClient) -> None:
        """Test invalid pagination returns 422."""
        response = await async_client.get(
            "/api/v1/documents",
            params={"page": 0}  # Invalid: must be >= 1
        )

        assert response.status_code == 422

    async def test_page_size_too_large(self, async_client: AsyncClient) -> None:
        """Test page size > 100 returns 422."""
        response = await async_client.get(
            "/api/v1/documents",
            params={"page_size": 200}  # Invalid: max 100
        )

        assert response.status_code == 422
