#!/usr/bin/env python
"""
Quick inline API verification using TestClient.

This script tests API endpoints WITHOUT requiring a running server.
It uses FastAPI's TestClient for direct testing.

Usage:
    python scripts/verify_api.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health():
    """Test health endpoint."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    return "✅ /health"


def test_root():
    """Test root endpoint."""
    r = client.get("/")
    assert r.status_code == 200
    assert "LIFE AI" in r.json()["service"]
    return "✅ /"


def test_openapi():
    """Test OpenAPI schema."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert len(paths) > 10  # Should have many endpoints
    return f"✅ /openapi.json ({len(paths)} paths)"


def test_documents_list():
    """Test document listing."""
    r = client.get("/api/v1/documents")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert "total" in data
    return f"✅ GET /documents ({data['total']} total)"


def test_entities_types():
    """Test entity types endpoint."""
    r = client.get("/api/v1/entities/types")
    assert r.status_code == 200
    types = r.json()
    summary = ", ".join([f"{t['type']}:{t['count']}" for t in types[:3]]) if types else "none"
    return f"✅ GET /entities/types ({summary})"


def test_entities_search():
    """Test entity search."""
    r = client.get("/api/v1/entities/search", params={"query": "parkinson"})
    assert r.status_code == 200
    data = r.json()
    return f"✅ GET /entities/search ({data['total']} found for 'parkinson')"


def test_graph_stats():
    """Test graph stats."""
    r = client.get("/api/v1/graph/stats")
    assert r.status_code == 200
    stats = r.json()
    return f"✅ GET /graph/stats (entities:{stats['entity_count']}, relations:{stats['relation_count']})"


def test_jobs_list():
    """Test jobs listing."""
    r = client.get("/api/v1/jobs")
    assert r.status_code == 200
    data = r.json()
    return f"✅ GET /jobs ({data['total']} jobs)"


def test_jobs_progress():
    """Test extraction progress."""
    r = client.get("/api/v1/jobs/progress/current")
    assert r.status_code == 200
    data = r.json()
    return f"✅ GET /jobs/progress ({data['chunks_processed']}/{data['chunks_total']} chunks)"


def test_entity_detail():
    """Test getting entity detail and neighborhood."""
    # First search for an entity
    r = client.get("/api/v1/entities/search", params={"query": "a", "page_size": 1})
    if r.status_code != 200 or not r.json()["items"]:
        return "⚠️ SKIP entity detail (no entities)"
    
    entity_id = r.json()["items"][0]["id"]
    entity_name = r.json()["items"][0]["name"]
    
    # Get detail
    r = client.get(f"/api/v1/entities/{entity_id}")
    assert r.status_code == 200
    
    # Get neighborhood
    r = client.get(f"/api/v1/entities/{entity_id}/neighborhood")
    assert r.status_code == 200
    rel_count = r.json()["total_relations"]
    
    return f"✅ GET /entities/{{id}} + neighborhood ({entity_name}, {rel_count} relations)"


def test_subgraph():
    """Test subgraph retrieval."""
    # Get an entity
    r = client.get("/api/v1/entities/search", params={"query": "a", "page_size": 1})
    if r.status_code != 200 or not r.json()["items"]:
        return "⚠️ SKIP subgraph (no entities)"
    
    entity_id = r.json()["items"][0]["id"]
    
    r = client.get("/api/v1/graph/subgraph", params={
        "entity_id": entity_id,
        "depth": 1,
        "max_nodes": 20
    })
    assert r.status_code == 200
    stats = r.json()["stats"]
    return f"✅ GET /graph/subgraph ({stats['node_count']} nodes, {stats['edge_count']} edges)"


def test_path_finding():
    """Test path finding."""
    # Get two entities
    r = client.get("/api/v1/entities/search", params={"query": "a", "page_size": 2})
    if r.status_code != 200 or len(r.json()["items"]) < 2:
        return "⚠️ SKIP path finding (need 2+ entities)"
    
    id1 = r.json()["items"][0]["id"]
    id2 = r.json()["items"][1]["id"]
    
    r = client.get("/api/v1/graph/path", params={
        "source_id": id1,
        "target_id": id2,
        "max_hops": 3
    })
    assert r.status_code == 200
    path_count = r.json()["path_count"]
    return f"✅ GET /graph/path ({path_count} paths found)"


def main():
    """Run all tests."""
    print()
    print("=" * 60)
    print("LIFE AI Knowledge Graph API - Quick Verification")
    print("=" * 60)
    print()
    
    tests = [
        test_health,
        test_root,
        test_openapi,
        test_documents_list,
        test_entities_types,
        test_entities_search,
        test_entity_detail,
        test_graph_stats,
        test_subgraph,
        test_path_finding,
        test_jobs_list,
        test_jobs_progress,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            result = test()
            print(f"  {result}")
            if "✅" in result:
                passed += 1
            elif "⚠️" in result:
                skipped += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    
    print()
    print("-" * 60)
    print(f"  Total: {len(tests)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("=" * 60)
    print()
    
    if failed == 0:
        print("🎉 All API endpoints working correctly!")
    else:
        print(f"⚠️ {failed} test(s) failed - check the errors above")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
