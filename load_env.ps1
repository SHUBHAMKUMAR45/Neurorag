# ─── Load .env into current PowerShell session ────────────────────────────────
# Usage: . .\load_env.ps1   (note the dot-space prefix — required for scope)

if (-not (Test-Path ".env")) {
    Write-Host "No .env file found. Create one from .env.example" -ForegroundColor Red
    return
}

$count = 0
Get-Content ".env" | Where-Object { $_ -match "^\s*[A-Z]" -and $_ -notmatch "^\s*#" } | ForEach-Object {
    $parts = $_ -split "=", 2
    if ($parts.Count -eq 2) {
        $name  = $parts[0].Trim()
        $value = $parts[1].Trim()
        [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        $count++
    }
}
Write-Host "Loaded $count environment variables from .env" -ForegroundColor Green
Write-Host "  OPENAI_API_KEY  : $( if ($env:OPENAI_API_KEY)  { 'SET ✓' } else { 'MISSING ✗' } )"
Write-Host "  NEURORAG_API_KEY: $( if ($env:NEURORAG_API_KEY) { 'SET ✓' } else { 'MISSING ✗' } )"
Write-Host "  POSTGRES_URL    : $( if ($env:POSTGRES_URL)     { 'SET ✓' } else { 'MISSING ✗' } )"
Write-Host "  REDIS_URL       : $( if ($env:REDIS_URL)        { 'SET ✓' } else { 'MISSING ✗' } )"
