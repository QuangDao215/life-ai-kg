#!/usr/bin/env python
"""
Quick API smoke test script.

This script verifies that all Phase 4 API endpoints are working correctly
by making actual HTTP requests to the running server.

Usage:
    1. Start the server: uvicorn src.main:app --reload --port 8000
    2. Run this script: python scripts/test_api.py

This is NOT a replacement for proper pytest tests (Phase 5),
just a quick sanity check.
"""

import asyncio
import sys

import httpx

BASE_URL = "http://localhost:8000"


async def test_api():
    """Run quick smoke tests on all API endpoints."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        results = []
        
        print("=" * 60)
        print("LIFE AI Knowledge Graph API - Smoke Test")
        print("=" * 60)
        print()

        # =================================================================
        # Health & Root
        # =================================================================
        print("Testing Health & Root endpoints...")
        
        # Health check
        try:
            r = await client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "healthy"
            results.append(("GET /health", "✅ PASS"))
        except Exception as e:
            results.append(("GET /health", f"❌ FAIL: {e}"))

        # Root
        try:
            r = await client.get("/")
            assert r.status_code == 200
            assert "LIFE AI" in r.json()["service"]
            results.append(("GET /", "✅ PASS"))
        except Exception as e:
            results.append(("GET /", f"❌ FAIL: {e}"))

        # =================================================================
        # Documents API
        # =================================================================
        print("Testing Documents API...")

        # List documents
        try:
            r = await client.get("/api/v1/documents")
            assert r.status_code == 200
            data = r.json()
            assert "items" in data
            assert "total" in data
            doc_count = data["total"]
            results.append(("GET /documents", f"✅ PASS ({doc_count} docs)"))
            
            # Get first document if exists
            if data["items"]:
                doc_id = data["items"][0]["id"]
                
                # Get single document
                r = await client.get(f"/api/v1/documents/{doc_id}")
                assert r.status_code == 200
                results.append(("GET /documents/{id}", "✅ PASS"))
                
                # Get document with chunks
                r = await client.get(f"/api/v1/documents/{doc_id}/chunks")
                assert r.status_code == 200
                chunk_count = len(r.json().get("chunks", []))
                results.append(("GET /documents/{id}/chunks", f"✅ PASS ({chunk_count} chunks)"))
        except Exception as e:
            results.append(("Documents API", f"❌ FAIL: {e}"))

        # =================================================================
        # Entities API
        # =================================================================
        print("Testing Entities API...")

        # Get entity types
        try:
            r = await client.get("/api/v1/entities/types")
            assert r.status_code == 200
            types = r.json()
            type_summary = ", ".join([f"{t['type']}:{t['count']}" for t in types[:3]])
            results.append(("GET /entities/types", f"✅ PASS ({type_summary}...)"))
        except Exception as e:
            results.append(("GET /entities/types", f"❌ FAIL: {e}"))

        # Search entities
        try:
            r = await client.get("/api/v1/entities/search", params={"query": "parkinson"})
            assert r.status_code == 200
            data = r.json()
            entity_count = data["total"]
            results.append(("GET /entities/search?query=parkinson", f"✅ PASS ({entity_count} found)"))
            
            # Get first entity if exists
            if data["items"]:
                entity_id = data["items"][0]["id"]
                
                # Get entity detail
                r = await client.get(f"/api/v1/entities/{entity_id}")
                assert r.status_code == 200
                entity_name = r.json()["name"]
                results.append(("GET /entities/{id}", f"✅ PASS ({entity_name})"))
                
                # Get entity neighborhood
                r = await client.get(f"/api/v1/entities/{entity_id}/neighborhood")
                assert r.status_code == 200
                rel_count = r.json()["total_relations"]
                results.append(("GET /entities/{id}/neighborhood", f"✅ PASS ({rel_count} relations)"))
                
                # Get entity relations
                r = await client.get(f"/api/v1/entities/{entity_id}/relations")
                assert r.status_code == 200
                results.append(("GET /entities/{id}/relations", "✅ PASS"))
        except Exception as e:
            results.append(("Entities API", f"❌ FAIL: {e}"))

        # =================================================================
        # Graph API
        # =================================================================
        print("Testing Graph API...")

        # Graph stats
        try:
            r = await client.get("/api/v1/graph/stats")
            assert r.status_code == 200
            stats = r.json()
            results.append((
                "GET /graph/stats", 
                f"✅ PASS (entities:{stats['entity_count']}, relations:{stats['relation_count']})"
            ))
        except Exception as e:
            results.append(("GET /graph/stats", f"❌ FAIL: {e}"))

        # Subgraph (need an entity ID)
        try:
            # Get an entity first
            r = await client.get("/api/v1/entities/search", params={"query": "a", "page_size": 1})
            if r.status_code == 200 and r.json()["items"]:
                entity_id = r.json()["items"][0]["id"]
                
                r = await client.get("/api/v1/graph/subgraph", params={
                    "entity_id": entity_id,
                    "depth": 1,
                    "max_nodes": 20
                })
                assert r.status_code == 200
                data = r.json()
                results.append((
                    "GET /graph/subgraph",
                    f"✅ PASS ({data['stats']['node_count']} nodes, {data['stats']['edge_count']} edges)"
                ))
        except Exception as e:
            results.append(("GET /graph/subgraph", f"❌ FAIL: {e}"))

        # Path finding (need two entity IDs)
        try:
            r = await client.get("/api/v1/entities/search", params={"query": "a", "page_size": 2})
            if r.status_code == 200 and len(r.json()["items"]) >= 2:
                id1 = r.json()["items"][0]["id"]
                id2 = r.json()["items"][1]["id"]
                
                r = await client.get("/api/v1/graph/path", params={
                    "source_id": id1,
                    "target_id": id2,
                    "max_hops": 3
                })
                assert r.status_code == 200
                data = r.json()
                results.append((
                    "GET /graph/path",
                    f"✅ PASS ({data['path_count']} paths found)"
                ))
            else:
                results.append(("GET /graph/path", "⚠️ SKIP (need 2+ entities)"))
        except Exception as e:
            results.append(("GET /graph/path", f"❌ FAIL: {e}"))

        # =================================================================
        # Jobs API
        # =================================================================
        print("Testing Jobs API...")

        # List jobs
        try:
            r = await client.get("/api/v1/jobs")
            assert r.status_code == 200
            job_count = r.json()["total"]
            results.append(("GET /jobs", f"✅ PASS ({job_count} jobs)"))
        except Exception as e:
            results.append(("GET /jobs", f"❌ FAIL: {e}"))

        # Get extraction progress
        try:
            r = await client.get("/api/v1/jobs/progress/current")
            assert r.status_code == 200
            data = r.json()
            results.append((
                "GET /jobs/progress/current",
                f"✅ PASS ({data['chunks_processed']}/{data['chunks_total']} chunks)"
            ))
        except Exception as e:
            results.append(("GET /jobs/progress/current", f"❌ FAIL: {e}"))

        # =================================================================
        # OpenAPI docs
        # =================================================================
        print("Testing OpenAPI docs...")
        
        try:
            r = await client.get("/openapi.json")
            assert r.status_code == 200
            assert "paths" in r.json()
            path_count = len(r.json()["paths"])
            results.append(("GET /openapi.json", f"✅ PASS ({path_count} paths documented)"))
        except Exception as e:
            results.append(("GET /openapi.json", f"❌ FAIL: {e}"))

        # =================================================================
        # Summary
        # =================================================================
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        passed = 0
        failed = 0
        skipped = 0
        
        for endpoint, status in results:
            print(f"  {status:50} {endpoint}")
            if "✅" in status:
                passed += 1
            elif "❌" in status:
                failed += 1
            else:
                skipped += 1
        
        print()
        print("-" * 60)
        print(f"  Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
        print("=" * 60)
        
        return failed == 0


async def main():
    """Main entry point."""
    print()
    print("Connecting to API at", BASE_URL)
    print()
    
    try:
        success = await test_api()
        sys.exit(0 if success else 1)
    except httpx.ConnectError:
        print("❌ ERROR: Cannot connect to API server!")
        print()
        print("Make sure the server is running:")
        print("  uvicorn src.main:app --reload --port 8000")
        print()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
