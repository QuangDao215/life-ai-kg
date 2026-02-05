# ============================================
# LIFE AI Knowledge Graph - Windows Setup Script
# ============================================
# Usage: .\setup.ps1
# 
# This script automates the complete project setup:
# 1. Creates Python virtual environment
# 2. Installs dependencies
# 3. Sets up environment variables
# 4. Starts Docker services (PostgreSQL + Redis)
# 5. Runs database migrations
# 6. Verifies the setup
# ============================================

param(
    [switch]$SkipDocker,
    [switch]$SkipVenv,
    [switch]$Help
)

# Colors for output
function Write-Step { param($msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Help message
if ($Help) {
    Write-Host @"
LIFE AI Knowledge Graph - Setup Script

Usage: .\setup.ps1 [options]

Options:
    -SkipDocker    Skip Docker container startup (if already running)
    -SkipVenv      Skip virtual environment creation (if already exists)
    -Help          Show this help message

Prerequisites:
    - Python 3.11+
    - Docker Desktop (running)
    - Git (optional)

"@
    exit 0
}

Write-Host @"
============================================
 LIFE AI Knowledge Graph - Setup
============================================
"@ -ForegroundColor Magenta

# ============================================
# Step 1: Check Prerequisites
# ============================================
Write-Step "Checking prerequisites..."

# Check Python
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python is not installed or not in PATH"
    Write-Host "Please install Python 3.11+ from https://www.python.org/downloads/"
    exit 1
}
Write-Success "Python found: $pythonVersion"

# Check Python version is 3.11+
$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
if ($Matches[1] -lt 3 -or ($Matches[1] -eq 3 -and $Matches[2] -lt 11)) {
    Write-Error "Python 3.11+ is required. Found: $pythonVersion"
    exit 1
}

# Check Docker
if (-not $SkipDocker) {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker is not installed or not in PATH"
        Write-Host "Please install Docker Desktop from https://www.docker.com/products/docker-desktop/"
        exit 1
    }
    Write-Success "Docker found: $dockerVersion"
    
    # Check if Docker daemon is running
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    }
    Write-Success "Docker daemon is running"
}

# ============================================
# Step 2: Create Virtual Environment
# ============================================
if (-not $SkipVenv) {
    Write-Step "Creating Python virtual environment..."
    
    if (Test-Path "venv") {
        Write-Warning "Virtual environment already exists. Skipping creation."
    } else {
        python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create virtual environment"
            exit 1
        }
        Write-Success "Virtual environment created"
    }
}

# ============================================
# Step 3: Activate Virtual Environment
# ============================================
Write-Step "Activating virtual environment..."

# Check execution policy
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted") {
    Write-Warning "Execution policy is Restricted. Attempting to change..."
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
}

# Activate venv
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to activate virtual environment"
    Write-Host "Try running: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    exit 1
}
Write-Success "Virtual environment activated"

# ============================================
# Step 4: Install Dependencies
# ============================================
Write-Step "Installing Python dependencies..."

python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upgrade pip"
    exit 1
}

pip install -e ".[dev]" --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install dependencies"
    exit 1
}
Write-Success "Dependencies installed"

# ============================================
# Step 5: Setup Environment Variables
# ============================================
Write-Step "Setting up environment variables..."

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Success "Created .env from .env.example"
        Write-Warning "Edit .env to set GOOGLE_API_KEY before running the demo"
    } else {
        Write-Error ".env.example not found! Cannot create .env"
        Write-Host "  Please ensure .env.example is present in the project root." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Warning ".env already exists. Skipping."
}

# ============================================
# Step 6: Start Docker Services
# ============================================
if (-not $SkipDocker) {
    Write-Step "Starting Docker services (PostgreSQL + Redis)..."
    
    docker-compose up -d postgres redis
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to start Docker services"
        exit 1
    }
    
    # Wait for services to be healthy
    Write-Host "Waiting for services to be healthy..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
    
    # Check PostgreSQL
    $retries = 0
    $maxRetries = 30
    while ($retries -lt $maxRetries) {
        $pgReady = docker-compose exec -T postgres pg_isready -U lifeai -d lifeai_kg 2>&1
        if ($LASTEXITCODE -eq 0) {
            break
        }
        $retries++
        Start-Sleep -Seconds 1
    }
    
    if ($retries -eq $maxRetries) {
        Write-Error "PostgreSQL failed to start within timeout"
        exit 1
    }
    Write-Success "PostgreSQL is ready"
    
    # Check Redis
    $redisReady = docker-compose exec -T redis redis-cli ping 2>&1
    if ($redisReady -match "PONG") {
        Write-Success "Redis is ready"
    } else {
        Write-Error "Redis is not responding"
        exit 1
    }
}

# ============================================
# Step 7: Run Database Migrations
# ============================================
Write-Step "Running database migrations..."

alembic upgrade head
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Migrations failed (this is expected if models are not yet defined)"
} else {
    Write-Success "Database migrations applied"
}

# ============================================
# Step 8: Verify Setup
# ============================================
Write-Step "Verifying setup..."

# Auto-fix lint issues
python -m ruff check src/ --fix --quiet 2>&1 | Out-Null
python -m ruff check src/ --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Success "Linting passed"
} else {
    Write-Warning "Linting has issues (run 'ruff check src/' for details)"
}

# Run unit tests (no DB required; integration tests run via run-demo.ps1)
$testOutput = pytest tests/unit/ -q 2>&1
if ($LASTEXITCODE -eq 0) {
    $passLine = $testOutput | Select-String "passed" | Select-Object -Last 1
    Write-Success "Unit tests passed ($passLine)"
} else {
    Write-Warning "Some unit tests failed (run 'pytest tests/unit/ -v' for details)"
}

# ============================================
# Done!
# ============================================
Write-Host @"

============================================
 Setup Complete!
============================================
"@ -ForegroundColor Green

Write-Host @"
Next steps:

  1. Set your Google API key:
     Open .env and set GOOGLE_API_KEY=your_key_here

  2. Run the full end-to-end demo:
     .\run-demo.ps1

  This will start the API server, Celery worker, fetch PubMed
  articles, build the knowledge graph, run queries, generate
  the evaluation report, and produce an interactive visualization.

  Or start services manually:
     uvicorn src.main:app --reload --port 8000
     celery -A src.workers.celery_app worker -l info -P solo -Q kg_extraction

  API docs: http://localhost:8000/docs

"@ -ForegroundColor White
