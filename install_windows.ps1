# ─── NeuroRAG Windows Setup Script ────────────────────────────────────────────
# Run this from the neurorag_final folder:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\install_windows.ps1

Write-Host "`n=== NeuroRAG Windows Setup ===" -ForegroundColor Cyan

# Step 1: Downgrade numpy first (faiss-cpu 1.8.0 requires numpy < 2)
Write-Host "`n[1/6] Pinning numpy to 1.26.4 (required for faiss-cpu compatibility)..." -ForegroundColor Yellow
pip install "numpy==1.26.4" --force-reinstall --quiet
if ($LASTEXITCODE -ne 0) { Write-Host "numpy install failed" -ForegroundColor Red; exit 1 }

# Step 2: Install faiss-cpu (no GPU needed)
Write-Host "[2/6] Installing faiss-cpu..." -ForegroundColor Yellow
pip install faiss-cpu --quiet
if ($LASTEXITCODE -ne 0) { Write-Host "faiss-cpu install failed" -ForegroundColor Red; exit 1 }

# Step 3: Install packaging (required by faiss loader)
Write-Host "[3/6] Installing packaging..." -ForegroundColor Yellow
pip install packaging --quiet

# Step 4: Install core requirements (skip problematic packages)
Write-Host "[4/6] Installing core requirements..." -ForegroundColor Yellow
pip install `
  fastapi==0.111.0 `
  "uvicorn[standard]==0.29.0" `
  pydantic==2.7.1 `
  pydantic-settings==2.3.0 `
  openai==1.30.1 `
  "sentence-transformers==3.0.1" `
  transformers==4.41.1 `
  accelerate==0.30.0 `
  whoosh==2.7.4 `
  asyncpg==0.29.0 `
  sqlalchemy==2.0.30 `
  "alembic==1.13.1" `
  psycopg2-binary==2.9.9 `
  "redis[asyncio]==5.0.4" `
  prometheus-client==0.20.0 `
  "structlog==24.1.0" `
  pyyaml==6.0.1 `
  python-dotenv==1.0.1 `
  packaging `
  httpx==0.27.0 `
  requests==2.32.3 `
  aiofiles==23.2.1 `
  tenacity==8.3.0 `
  cryptography==42.0.5 `
  "opentelemetry-api==1.24.0" `
  "opentelemetry-sdk==1.24.0" `
  --quiet

if ($LASTEXITCODE -ne 0) { Write-Host "Core requirements install failed" -ForegroundColor Red; exit 1 }

# Step 5: Create data directories
Write-Host "[5/6] Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\faiss" | Out-Null
New-Item -ItemType Directory -Force -Path "data\whoosh_index" | Out-Null
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "static" | Out-Null

# Step 6: Check .env
Write-Host "[6/6] Checking .env file..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  Created .env from .env.example — fill in OPENAI_API_KEY and NEURORAG_API_KEY" -ForegroundColor Yellow
} else {
    Write-Host "  .env already exists" -ForegroundColor Green
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env and set OPENAI_API_KEY and NEURORAG_API_KEY"
Write-Host "  2. Set env vars:  . .\load_env.ps1"
Write-Host "  3. Run migrations: alembic upgrade head"
Write-Host "  4. Start API:      uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1"
Write-Host "  5. Open browser:   http://localhost:8000"
Write-Host "  6. Seed data:      python scripts\seed_data.py"
