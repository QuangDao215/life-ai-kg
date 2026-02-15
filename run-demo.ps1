# ============================================
# LIFE AI Knowledge Graph - End-to-End Demo
# ============================================
#
# This script runs the COMPLETE pipeline:
#   1. Pre-flight checks (venv, Docker, API, Celery worker)
#   2. PubMed fetch -> documents in database
#   3. Chunk documents -> text chunks with offsets
#   4. KG build via API -> Celery queue -> worker extraction
#   5. Poll job status (shows pending -> running -> completed)
#   6. Demo API queries -> saved JSON responses
#   7. Evaluation -> EVAL_REPORT.md
#   8. Visualization -> interactive HTML graph
#
# Prerequisites: Run .\setup.ps1 first
#
# Usage:
#   .\run-demo.ps1                    # Full demo with Gemini
#   .\run-demo.ps1 -Provider mock     # Use mock LLM (no API key needed)
#   .\run-demo.ps1 -SkipFetch         # Skip PubMed fetch (reuse data)
#   .\run-demo.ps1 -MaxDocuments 50   # Fetch more documents
#   .\run-demo.ps1 -ChunkLimit 10     # Only extract 10 chunks
#   .\run-demo.ps1 -ChunkSize 500 -ChunkOverlap 100  # Smaller chunks
# ============================================

param(
    [int]$MaxDocuments = 20,
    [int]$ChunkLimit = 20,
    [int]$ChunkSize = 1000,
    [int]$ChunkOverlap = 200,
    [string]$Provider = "gemini",
    [switch]$SkipFetch,
    [switch]$SkipChunk,
    [switch]$SkipEval,
    [switch]$Help
)

# ============================================
# Helpers
# ============================================
function Write-Step { param($msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK { param($msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "  [ERROR] $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "  $msg" -ForegroundColor Gray }

if ($Help) {
    Write-Host @"
LIFE AI Knowledge Graph - End-to-End Demo

Usage: .\run-demo.ps1 [options]

Options:
    -MaxDocuments <n>   PubMed documents to fetch (default: 20)
    -ChunkLimit <n>     Max chunks to extract, 0 = all (default: 20)
    -ChunkSize <n>      Characters per chunk (default: 1000)
    -ChunkOverlap <n>   Overlap between chunks (default: 200)
    -Provider <name>    LLM provider: gemini or mock (default: gemini)
    -SkipFetch          Skip PubMed fetch step
    -SkipChunk          Skip chunking step
    -SkipEval           Skip evaluation step
    -Help               Show this message

Pipeline:
    PubMed Fetch -> Chunking -> Celery KG Build -> API Queries -> Evaluation -> Visualization
"@
    exit 0
}

$ErrorActionPreference = "Stop"
$apiBase = "http://localhost:8000"
$outputDir = "demo_output"

Write-Host @"

============================================================
  LIFE AI Knowledge Graph - End-to-End Demo
============================================================
  Provider:      $Provider
  Max documents: $MaxDocuments
  Chunk limit:   $(if ($ChunkLimit -eq 0) { "all" } else { $ChunkLimit })
  Chunk size:    $ChunkSize chars (overlap: $ChunkOverlap)
============================================================
"@ -ForegroundColor Magenta


# ============================================
# Step 0: Pre-flight Checks
# ============================================
Write-Step "STEP 0: Pre-flight checks"

# Venv
if (-not $env:VIRTUAL_ENV) {
    Write-Warn "Virtual environment not activated. Activating..."
    if (Test-Path ".\venv\Scripts\Activate.ps1") {
        & .\venv\Scripts\Activate.ps1
    } else {
        Write-Err "Virtual environment not found. Run .\setup.ps1 first."
        exit 1
    }
}
Write-OK "Virtual environment active"

# Docker
$pgRunning = docker-compose ps postgres 2>&1 | Select-String "running"
$redisRunning = docker-compose ps redis 2>&1 | Select-String "running"
if (-not $pgRunning -or -not $redisRunning) {
    Write-Warn "Docker services not running. Starting..."
    docker-compose up -d postgres redis
    Start-Sleep -Seconds 5
}
Write-OK "Docker services running (PostgreSQL + Redis)"

# API server
$apiRunning = $false
try {
    $health = Invoke-RestMethod -Uri "$apiBase/health" -Method Get -TimeoutSec 2
    $apiRunning = $true
    Write-OK "API server running"
} catch {
    Write-Warn "API server not running. Starting..."
    Start-Process -FilePath "uvicorn" -ArgumentList "src.main:app", "--port", "8000" -WindowStyle Hidden
    Start-Sleep -Seconds 4
    try {
        $health = Invoke-RestMethod -Uri "$apiBase/health" -Method Get -TimeoutSec 5
        $apiRunning = $true
        Write-OK "API server started"
    } catch {
        Write-Err "Could not start API server. Start manually: uvicorn src.main:app --port 8000"
        exit 1
    }
}

# Celery worker
Write-Info "Starting Celery worker..."
$celeryProcess = Start-Process -FilePath "celery" `
    -ArgumentList "-A", "src.workers.celery_app", "worker", "-l", "info", "-P", "solo", "-Q", "kg_extraction" `
    -WindowStyle Hidden `
    -PassThru
Start-Sleep -Seconds 3

if ($celeryProcess -and -not $celeryProcess.HasExited) {
    Write-OK "Celery worker started (PID: $($celeryProcess.Id))"
} else {
    Write-Warn "Celery worker may not have started. KG build will use in-process fallback."
}


# ============================================
# Step 1: Fetch PubMed Data
# ============================================
if (-not $SkipFetch) {
    Write-Step "STEP 1: Fetching PubMed publications"
    Write-Info "Query: 'ambroxol parkinson', max results: $MaxDocuments"

    python scripts/fetch_pubmed.py --max-results $MaxDocuments
    if ($LASTEXITCODE -ne 0) {
        Write-Err "PubMed fetch failed"
        exit 1
    }
    Write-OK "PubMed data fetched"
} else {
    Write-Step "STEP 1: Skipping PubMed fetch (--SkipFetch)"
}


# ============================================
# Step 2: Chunk Documents
# ============================================
if (-not $SkipChunk) {
    Write-Step "STEP 2: Chunking documents"
    Write-Info "Splitting documents into text chunks (size=$ChunkSize, overlap=$ChunkOverlap)..."

    python scripts/chunk_documents.py --chunk-size $ChunkSize --overlap $ChunkOverlap
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Chunking failed"
        exit 1
    }
    Write-OK "Documents chunked"
} else {
    Write-Step "STEP 2: Skipping chunking (--SkipChunk)"
}


# ============================================
# Step 3: Trigger KG Build via API (Celery Queue)
# ============================================
Write-Step "STEP 3: Triggering KG build via API"

$body = @{
    provider = $Provider
    skip_processed = $true
} | ConvertTo-Json

if ($ChunkLimit -gt 0) {
    $body = @{
        provider = $Provider
        skip_processed = $true
        limit = $ChunkLimit
    } | ConvertTo-Json
}

Write-Info "POST /api/v1/jobs/kg-build"
Write-Info "Body: $body"

try {
    $buildResponse = Invoke-RestMethod `
        -Uri "$apiBase/api/v1/jobs/kg-build" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body
    
    $jobId = $buildResponse.id

    Write-Host ""
    Write-Host "  Response (202 Accepted):" -ForegroundColor Yellow
    Write-Host "    id:       $jobId" -ForegroundColor White
    Write-Host "    status:   $($buildResponse.status)" -ForegroundColor White
    Write-Host "    message:  $($buildResponse.message)" -ForegroundColor White
    Write-Host ""
    Write-OK "Job created: $jobId"
} catch {
    Write-Err "Failed to trigger KG build: $_"
    exit 1
}


# ============================================
# Step 4: Poll Job Status (shows progression)
# ============================================
Write-Step "STEP 4: Polling job status"
Write-Info "GET /api/v1/jobs/$jobId"
Write-Host ""

$maxWaitSeconds = 20000
$waitedSeconds = 0
$pollInterval = 3
$lastStatus = ""

while ($waitedSeconds -lt $maxWaitSeconds) {
    try {
        $jobStatus = Invoke-RestMethod -Uri "$apiBase/api/v1/jobs/$jobId" -Method Get
        $currentStatus = $jobStatus.status
        $progress = [math]::Round($jobStatus.progress, 1)
        $processed = $jobStatus.processed_items
        $total = $jobStatus.total_items

        # Print status change clearly
        if ($currentStatus -ne $lastStatus) {
            Write-Host ""
            Write-Host "  >>> Status changed: $lastStatus -> $currentStatus" -ForegroundColor Yellow
            $lastStatus = $currentStatus
        }

        # Progress bar
        $bar = ""
        $barLen = 30
        $filled = [math]::Floor($progress / 100 * $barLen)
        $bar = ("=" * $filled) + ("-" * ($barLen - $filled))
        Write-Host "`r  [$bar] $progress% ($processed/$total chunks) | Status: $currentStatus    " -NoNewline

        if ($currentStatus -eq "completed") {
            Write-Host ""
            Write-Host ""
            Write-Host "  Final job result:" -ForegroundColor Yellow
            $jobStatus.result | ConvertTo-Json | Write-Host -ForegroundColor White
            Write-Host ""
            Write-OK "KG build completed!"
            break
        }
        elseif ($currentStatus -eq "failed") {
            Write-Host ""
            Write-Err "KG build failed: $($jobStatus.error_message)"
            Write-Info "Check Celery worker terminal for full traceback."
            exit 1
        }

        Start-Sleep -Seconds $pollInterval
        $waitedSeconds += $pollInterval
    } catch {
        Write-Warn "Poll failed, retrying..."
        Start-Sleep -Seconds $pollInterval
        $waitedSeconds += $pollInterval
    }
}

if ($waitedSeconds -ge $maxWaitSeconds) {
    Write-Err "Timeout after ${maxWaitSeconds}s waiting for KG build"
    exit 1
}


# ============================================
# Step 5: Demo API Queries
# ============================================
Write-Step "STEP 5: Demonstrating graph exploration API"

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Also save to examples/api_responses/ for repo submission
$examplesDir = "examples/api_responses"
if (-not (Test-Path $examplesDir)) {
    New-Item -ItemType Directory -Path $examplesDir -Force | Out-Null
}

# Save the completed job status as an example
Write-Info "Saving job status example..."
try {
    $finalJobStatus = Invoke-RestMethod -Uri "$apiBase/api/v1/jobs/$jobId" -Method Get
    $finalJobStatus | ConvertTo-Json -Depth 10 | Out-File "$outputDir/job_status_completed.json" -Encoding utf8
    $finalJobStatus | ConvertTo-Json -Depth 10 | Out-File "$examplesDir/job_status_completed.json" -Encoding utf8
    Write-OK "Job status saved"
} catch {
    Write-Warn "Failed to save job status: $_"
}

# Query 1: Search for Ambroxol
Write-Info "Query 1: GET /api/v1/entities/search?query=ambroxol"
$ambroxolId = $null
try {
    $ambroxolSearch = Invoke-RestMethod -Uri "$apiBase/api/v1/entities/search?query=ambroxol" -Method Get
    $ambroxolSearch | ConvertTo-Json -Depth 10 | Out-File "$outputDir/query1_ambroxol_search.json" -Encoding utf8
    $ambroxolSearch | ConvertTo-Json -Depth 10 | Out-File "$examplesDir/entity_search.json" -Encoding utf8
    Write-OK "Found $($ambroxolSearch.total) entities matching 'ambroxol'"

    if ($ambroxolSearch.items.Count -gt 0) {
        $ambroxolId = $ambroxolSearch.items[0].id
        Write-Info "  Ambroxol entity ID: $ambroxolId"
    }
} catch {
    Write-Warn "Entity search failed: $_"
}

# Query 2: Search for Parkinson's
Write-Info "Query 2: GET /api/v1/entities/search?query=parkinson"
try {
    $pdSearch = Invoke-RestMethod -Uri "$apiBase/api/v1/entities/search?query=parkinson" -Method Get
    $pdSearch | ConvertTo-Json -Depth 10 | Out-File "$outputDir/query2_parkinson_search.json" -Encoding utf8
    Write-OK "Found $($pdSearch.total) entities matching 'parkinson'"
} catch {
    Write-Warn "Entity search failed: $_"
}

# Query 3: Ambroxol neighborhood
if ($ambroxolId) {
    Write-Info "Query 3: GET /api/v1/entities/$ambroxolId/neighborhood"
    try {
        $neighborhood = Invoke-RestMethod -Uri "$apiBase/api/v1/entities/$ambroxolId/neighborhood" -Method Get
        $neighborhood | ConvertTo-Json -Depth 10 | Out-File "$outputDir/query3_ambroxol_neighborhood.json" -Encoding utf8
        $neighborhood | ConvertTo-Json -Depth 10 | Out-File "$examplesDir/entity_neighborhood.json" -Encoding utf8
        Write-OK "Neighborhood: $($neighborhood.total_relations) relations"
    } catch {
        Write-Warn "Neighborhood query failed: $_"
    }
}

# Query 4: Subgraph from Ambroxol
if ($ambroxolId) {
    Write-Info "Query 4: GET /api/v1/graph/subgraph?entity_id=$ambroxolId&depth=2&max_nodes=50"
    try {
        $subgraph = Invoke-RestMethod -Uri "$apiBase/api/v1/graph/subgraph?entity_id=$ambroxolId&depth=2&max_nodes=50" -Method Get
        $subgraph | ConvertTo-Json -Depth 10 | Out-File "$outputDir/query4_ambroxol_subgraph.json" -Encoding utf8
        $subgraph | ConvertTo-Json -Depth 10 | Out-File "$examplesDir/subgraph.json" -Encoding utf8
        Write-OK "Subgraph: $($subgraph.nodes.Count) nodes, $($subgraph.edges.Count) edges"
    } catch {
        Write-Warn "Subgraph query failed: $_"
    }
}

# Query 5: Path between Ambroxol and GCase
if ($ambroxolId) {
    Write-Info "Query 5: Finding path Ambroxol -> GCase"
    try {
        $gcaseSearch = Invoke-RestMethod -Uri "$apiBase/api/v1/entities/search?query=gcase" -Method Get
        if ($gcaseSearch.items.Count -gt 0) {
            $gcaseId = $gcaseSearch.items[0].id
            Write-Info "  GCase entity ID: $gcaseId"

            $pathResult = Invoke-RestMethod -Uri "$apiBase/api/v1/graph/path?source_id=$ambroxolId&target_id=$gcaseId&max_hops=4" -Method Get
            $pathResult | ConvertTo-Json -Depth 10 | Out-File "$outputDir/query5_ambroxol_gcase_path.json" -Encoding utf8
            $pathResult | ConvertTo-Json -Depth 10 | Out-File "$examplesDir/path_ambroxol_gcase.json" -Encoding utf8

            if ($pathResult.paths.Count -gt 0) {
                Write-OK "Path found: $($pathResult.paths[0].path_length) hops, confidence: $($pathResult.paths[0].total_confidence)"
            } else {
                Write-Warn "No path found between Ambroxol and GCase within 4 hops"
            }
        } else {
            Write-Warn "GCase entity not found"
        }
    } catch {
        Write-Warn "Path query failed: $_"
    }
}

# Query 6: Graph statistics
Write-Info "Query 6: GET /api/v1/graph/stats"
try {
    $stats = Invoke-RestMethod -Uri "$apiBase/api/v1/graph/stats" -Method Get
    $stats | ConvertTo-Json -Depth 10 | Out-File "$outputDir/query6_graph_stats.json" -Encoding utf8
    $stats | ConvertTo-Json -Depth 10 | Out-File "$examplesDir/graph_stats.json" -Encoding utf8
    Write-OK "Graph stats saved"
} catch {
    Write-Warn "Stats query failed: $_"
}


# ============================================
# Step 6: Evaluation
# ============================================
if (-not $SkipEval) {
    Write-Step "STEP 6: Running evaluation harness"

    python scripts/eval.py --output "$outputDir/EVAL_REPORT.md"
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Evaluation completed with warnings"
    } else {
        Write-OK "Evaluation report generated"
    }

    # Save a copy to examples/ for repo submission
    Copy-Item "$outputDir/EVAL_REPORT.md" "$examplesDir/../EVAL_REPORT.md" -Force
    Write-OK "Evaluation report also saved to examples/EVAL_REPORT.md"
} else {
    Write-Step "STEP 6: Skipping evaluation (--SkipEval)"
}


# ============================================
# Step 7: Generate Visualization
# ============================================
Write-Step "STEP 7: Generating interactive KG visualization"

try {
    python scripts/visualize.py `
        --output "$outputDir/kg_visualization.html" `
        --centers "ambroxol,parkinson" `
        --depth 2 `
        --max-nodes 100

    if (Test-Path "$outputDir/kg_visualization.html") {
        Copy-Item "$outputDir/kg_visualization.html" "$examplesDir/../kg_visualization.html" -Force
        Write-OK "Visualization saved to $outputDir/kg_visualization.html"
        Write-OK "Also saved to examples/kg_visualization.html"
        Write-Info "Open in browser to explore the knowledge graph interactively"
    } else {
        Write-Warn "Visualization file not created"
    }
} catch {
    Write-Warn "Visualization generation failed: $_"
}


# ============================================
# Cleanup: Stop Celery worker
# ============================================
if ($celeryProcess -and -not $celeryProcess.HasExited) {
    Write-Info "Stopping Celery worker (PID: $($celeryProcess.Id))..."
    Stop-Process -Id $celeryProcess.Id -Force -ErrorAction SilentlyContinue
    Write-OK "Celery worker stopped"
}


# ============================================
# Summary
# ============================================
Write-Host @"

============================================================
  Demo Complete!
============================================================
"@ -ForegroundColor Green

Write-Host "Output files saved to: .\$outputDir\" -ForegroundColor White
Write-Host ""
Write-Host "  Saved API responses (also in .\examples\api_responses\):" -ForegroundColor Cyan
Write-Host "    query1_ambroxol_search.json       Entity search" -ForegroundColor Gray
Write-Host "    query2_parkinson_search.json      Entity search" -ForegroundColor Gray
Write-Host "    query3_ambroxol_neighborhood.json Neighborhood view" -ForegroundColor Gray
Write-Host "    query4_ambroxol_subgraph.json     Bounded subgraph" -ForegroundColor Gray
Write-Host "    query5_ambroxol_gcase_path.json   Path computation" -ForegroundColor Gray
Write-Host "    query6_graph_stats.json           Graph statistics" -ForegroundColor Gray
Write-Host "    EVAL_REPORT.md                    Evaluation report" -ForegroundColor Gray
Write-Host "    kg_visualization.html             Interactive graph visualization" -ForegroundColor Gray
Write-Host ""
Write-Host "  Interactive exploration:" -ForegroundColor Cyan
Write-Host "    API docs:       http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "    Visualization:  python scripts/visualize.py" -ForegroundColor Gray
Write-Host ""
