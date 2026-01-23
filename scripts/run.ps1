Write-Host "🚀 Running Streamlit App..." -ForegroundColor Cyan

# Activate venv
if (!(Test-Path ".\venv")) {
    Write-Host "❌ venv not found. Run scripts/setup.ps1 first." -ForegroundColor Red
    exit 1
}

.\venv\Scripts\Activate.ps1

# Ensure repo root is in PYTHONPATH so "import src" works
$env:PYTHONPATH = (Get-Location).Path

# Run app
streamlit run app/main.py
