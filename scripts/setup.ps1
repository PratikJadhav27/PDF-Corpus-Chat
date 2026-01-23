Write-Host "Setting up PDF-Corpus-Chat..." -ForegroundColor Cyan

# 1) Create venv if missing
if (!(Test-Path ".\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# 2) Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# 3) Upgrade pip
Write-Host "⬆Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 4) Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# 5) Create .env if missing
if (!(Test-Path ".\.env")) {
    Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".\.env.example" ".\.env"
    Write-Host "Please edit .env and add your GOOGLE_API_KEY" -ForegroundColor Red
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Next: run the app using scripts/run.ps1" -ForegroundColor Cyan
