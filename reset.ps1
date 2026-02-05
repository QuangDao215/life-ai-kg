# ============================================
# LIFE AI Knowledge Graph - Reset Script
# ============================================
# Stops services and cleans project state at three levels:
#
# Usage:
#   .\reset.ps1              # Basic: stop processes + remove generated files
#   .\reset.ps1 -WipeDB      # + wipe database and Redis volumes
#   .\reset.ps1 -Full         # + remove venv/ and .env (fresh clone state)
#   .\reset.ps1 -Help         # Show help
#
# After reset:
#   .\reset.ps1              -> .\run-demo.ps1
#   .\reset.ps1 -WipeDB      -> alembic upgrade head -> .\run-demo.ps1
#   .\reset.ps1 -Full         -> .\setup.ps1 -> .\run-demo.ps1
# ============================================

param(
    [switch]$WipeDB,
    [switch]$Full,
    [switch]$Help
)

# -Full implies -WipeDB
if ($Full) { $WipeDB = $true }

function Write-Step { param($msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK { param($msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Info { param($msg) Write-Host "  [..] $msg" -ForegroundColor Gray }
function Write-Warn { param($msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

if ($Help) {
    Write-Host @"
LIFE AI Knowledge Graph - Reset Script

Three levels of reset:

  .\reset.ps1           Basic reset
                        - Stop Celery worker and Uvicorn server
                        - Stop Docker containers (keep volumes)
                        - Remove generated files (demo_output, examples, caches)

  .\reset.ps1 -WipeDB   Database reset (includes basic)
                        - Everything above
                        - Remove Docker volumes (PostgreSQL + Redis data)
                        - Next: docker-compose up -d -> alembic upgrade head

  .\reset.ps1 -Full     Full reset (includes database)
                        - Everything above
                        - Remove venv/ and .env
                        - Next: .\setup.ps1

After reset:
  Basic:     .\run-demo.ps1                              (data still there)
  WipeDB:    docker-compose up -d -> alembic upgrade head -> .\run-demo.ps1
  Full:      .\setup.ps1 -> edit .env -> .\run-demo.ps1
"@
    exit 0
}

# Determine mode label
if ($Full) { $modeLabel = "FULL (fresh clone state)" }
elseif ($WipeDB) { $modeLabel = "DATABASE (wipe DB + Redis volumes)" }
else { $modeLabel = "BASIC (keep data)" }

Write-Host @"

============================================================
  LIFE AI Knowledge Graph - Reset
============================================================
  Mode: $modeLabel
============================================================
"@ -ForegroundColor Magenta

# ============================================
# Step 1: Stop processes
# ============================================
Write-Step "Stopping running processes"

# Stop Celery workers
$celeryProcs = Get-Process -Name "celery" -ErrorAction SilentlyContinue
if ($celeryProcs) {
    $celeryProcs | Stop-Process -Force
    Write-OK "Celery worker(s) stopped"
} else {
    Write-Info "No Celery workers running"
}

# Stop Uvicorn
$uvicornProcs = Get-Process -Name "uvicorn" -ErrorAction SilentlyContinue
if ($uvicornProcs) {
    $uvicornProcs | Stop-Process -Force
    Write-OK "Uvicorn server stopped"
} else {
    Write-Info "No Uvicorn server running"
}

# Hint about python processes
$pythonProcs = Get-Process -Name "python" -ErrorAction SilentlyContinue
if ($pythonProcs) {
    Write-Info "Found $($pythonProcs.Count) Python process(es) - not killing (may be unrelated)"
    Write-Info "If the API is still running, stop it manually: taskkill /F /IM python.exe"
}

# ============================================
# Step 2: Stop Docker containers
# ============================================
Write-Step "Stopping Docker services"

$dockerAvailable = docker --version 2>&1
if ($LASTEXITCODE -eq 0) {
    if ($WipeDB) {
        docker-compose down -v 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-OK "Docker containers stopped, volumes REMOVED (DB + Redis wiped)"
        } else {
            Write-Warn "docker-compose down -v failed (containers may not exist)"
        }
    } else {
        docker-compose down 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-OK "Docker containers stopped (volumes preserved)"
        } else {
            Write-Warn "docker-compose down failed (containers may not exist)"
        }
    }
} else {
    Write-Info "Docker not available, skipping"
}

# ============================================
# Step 3: Remove generated files
# ============================================
Write-Step "Removing generated files"

# Demo output
if (Test-Path "demo_output") {
    Remove-Item -Recurse -Force "demo_output"
    Write-OK "Removed demo_output/"
} else {
    Write-Info "demo_output/ not found"
}

# Example API responses (JSON files only, keep README.md)
$exampleJsons = Get-ChildItem "examples/api_responses/*.json" -ErrorAction SilentlyContinue
if ($exampleJsons) {
    $exampleJsons | Remove-Item -Force
    Write-OK "Removed $($exampleJsons.Count) example API response(s)"
} else {
    Write-Info "No example API responses to remove"
}

# Eval report
if (Test-Path "examples/EVAL_REPORT.md") {
    Remove-Item -Force "examples/EVAL_REPORT.md"
    Write-OK "Removed examples/EVAL_REPORT.md"
} else {
    Write-Info "EVAL_REPORT.md not found"
}

# Visualization
if (Test-Path "examples/kg_visualization.html") {
    Remove-Item -Force "examples/kg_visualization.html"
    Write-OK "Removed examples/kg_visualization.html"
} else {
    Write-Info "kg_visualization.html not found"
}

# Standalone visualization (if generated outside demo)
if (Test-Path "kg_visualization.html") {
    Remove-Item -Force "kg_visualization.html"
    Write-OK "Removed kg_visualization.html (root)"
}

# Python caches
$caches = Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
if ($caches) {
    $caches | Remove-Item -Recurse -Force
    Write-OK "Removed $($caches.Count) __pycache__ directories"
} else {
    Write-Info "No __pycache__ to clean"
}

# .pytest_cache
if (Test-Path ".pytest_cache") {
    Remove-Item -Recurse -Force ".pytest_cache"
    Write-OK "Removed .pytest_cache/"
}

# .ruff_cache
if (Test-Path ".ruff_cache") {
    Remove-Item -Recurse -Force ".ruff_cache"
    Write-OK "Removed .ruff_cache/"
}

# ============================================
# Step 4: Full reset (optional)
# ============================================
if ($Full) {
    Write-Step "Full reset: removing venv and .env"

    if (Test-Path "venv") {
        Remove-Item -Recurse -Force "venv"
        Write-OK "Removed venv/"
    } else {
        Write-Info "venv/ not found"
    }

    if (Test-Path ".env") {
        Remove-Item -Force ".env"
        Write-OK "Removed .env"
    } else {
        Write-Info ".env not found"
    }
}

# ============================================
# Done
# ============================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Reset Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

if ($Full) {
    Write-Host "  Full reset done. To start fresh:" -ForegroundColor White
    Write-Host ""
    Write-Host "    .\setup.ps1" -ForegroundColor White
    Write-Host "    # Edit .env -> set GOOGLE_API_KEY=your_key" -ForegroundColor White
    Write-Host "    .\run-demo.ps1" -ForegroundColor White
} elseif ($WipeDB) {
    Write-Host "  Database wiped. To rebuild:" -ForegroundColor White
    Write-Host ""
    Write-Host "    docker-compose up -d postgres redis" -ForegroundColor White
    Write-Host "    alembic upgrade head" -ForegroundColor White
    Write-Host "    .\run-demo.ps1" -ForegroundColor White
} else {
    Write-Host "  Processes stopped, files cleaned. Data intact. To continue:" -ForegroundColor White
    Write-Host ""
    Write-Host "    .\run-demo.ps1" -ForegroundColor White
}
Write-Host ""
