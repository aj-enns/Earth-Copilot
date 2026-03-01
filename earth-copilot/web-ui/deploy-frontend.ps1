# Earth Copilot Frontend Deployment Script
# Deploys the React web UI to Azure Static Web Apps
# Auto-discovers resources from Azure subscription

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "",
    
    [Parameter(Mandatory=$false)]
    [string]$StaticWebAppName = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipBuild = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$ShowDetails = $false
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EARTH COPILOT FRONTEND DEPLOYMENT" -ForegroundColor Cyan
Write-Host "(Azure Static Web Apps)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory (web-ui folder)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "[Working Directory: $ScriptDir]" -ForegroundColor Yellow
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "$ScriptDir\package.json")) {
    Write-Host "[ERROR] package.json not found. Are you in the web-ui directory?" -ForegroundColor Red
    exit 1
}

# Check Azure CLI
Write-Host "[Checking Azure CLI]" -ForegroundColor Cyan
try {
    $azVersion = az version --output json | ConvertFrom-Json
    Write-Host "[OK] Azure CLI version: $($azVersion.'azure-cli')" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Azure CLI not found. Please install from https://aka.ms/installazurecliwindows" -ForegroundColor Red
    exit 1
}

# Check if logged in to Azure
Write-Host "[Checking Azure login]" -ForegroundColor Cyan
try {
    $account = az account show 2>$null | ConvertFrom-Json
    Write-Host "[OK] Logged in as: $($account.user.name)" -ForegroundColor Green
    Write-Host "   Subscription: $($account.name)" -ForegroundColor Gray
} catch {
    Write-Host "[ERROR] Not logged in to Azure. Running 'az login'..." -ForegroundColor Yellow
    az login
}

# ========================================
# AUTO-DISCOVER RESOURCES IF NOT PROVIDED
# ========================================
Write-Host ""
Write-Host "[Discovering Azure Resources]" -ForegroundColor Cyan

# Find resource group if not provided
if ([string]::IsNullOrEmpty($ResourceGroup)) {
    Write-Host "   Looking for Earth Copilot resource group..." -ForegroundColor Gray
    
    $groups = az group list --query "[?contains(name, 'earthcopilot') || contains(name, 'earth-copilot')].name" -o tsv 2>$null
    
    if ($groups) {
        $ResourceGroup = ($groups -split "`n")[0].Trim()
        Write-Host "[OK] Found resource group: $ResourceGroup" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Could not find Earth Copilot resource group." -ForegroundColor Red
        Write-Host "   Please specify -ResourceGroup parameter or create infrastructure first." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[OK] Using provided resource group: $ResourceGroup" -ForegroundColor Green
}

# Find or create Static Web App
if ([string]::IsNullOrEmpty($StaticWebAppName)) {
    Write-Host "   Looking for Static Web App in $ResourceGroup..." -ForegroundColor Gray
    
    $swaList = az staticwebapp list --resource-group $ResourceGroup --query "[0].name" -o tsv 2>$null
    
    if ($swaList) {
        $StaticWebAppName = $swaList.Trim()
        Write-Host "[OK] Found Static Web App: $StaticWebAppName" -ForegroundColor Green
    } else {
        # Create one
        $StaticWebAppName = "swa-earthcopilot"
        Write-Host "[INFO] No Static Web App found. Creating: $StaticWebAppName" -ForegroundColor Yellow
        $location = (az group show --name $ResourceGroup --query "location" -o tsv 2>$null).Trim()
        az staticwebapp create --name $StaticWebAppName --resource-group $ResourceGroup --location $location --sku Free -o none
        Write-Host "[OK] Created Static Web App: $StaticWebAppName" -ForegroundColor Green
    }
} else {
    Write-Host "[OK] Using provided Static Web App: $StaticWebAppName" -ForegroundColor Green
}

# Get deployment token
Write-Host ""
Write-Host "[Getting deployment token]" -ForegroundColor Cyan
$deploymentToken = az staticwebapp secrets list --name $StaticWebAppName --resource-group $ResourceGroup --query "properties.apiKey" -o tsv 2>$null
if ([string]::IsNullOrEmpty($deploymentToken)) {
    Write-Host "[ERROR] Could not retrieve deployment token for Static Web App '$StaticWebAppName'" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Deployment token retrieved" -ForegroundColor Green

# Get SWA URL
$swaHostname = az staticwebapp show --name $StaticWebAppName --resource-group $ResourceGroup --query "defaultHostname" -o tsv 2>$null

if (-not $SkipBuild) {
    # Install dependencies
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "STEP 1/3: Installing Dependencies" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Push-Location $ScriptDir
    try {
        npm install
        if ($LASTEXITCODE -ne 0) {
            throw "npm install failed"
        }
        Write-Host "[OK] Dependencies installed" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Failed to install dependencies: $_" -ForegroundColor Red
        Pop-Location
        exit 1
    }

    # Build the application
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "STEP 2/3: Building Production Bundle" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "This may take 2-3 minutes..." -ForegroundColor Yellow
    Write-Host ""
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) {
            throw "npm run build failed"
        }
        Write-Host ""
        Write-Host "[OK] Build completed successfully" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Build failed: $_" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Pop-Location

    # Verify dist folder exists
    if (-not (Test-Path "$ScriptDir\dist")) {
        Write-Host "[ERROR] dist folder not found after build" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "[SKIP] Skipping build (using existing dist folder)" -ForegroundColor Yellow
}

# Deploy to Azure Static Web Apps using SWA CLI
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STEP 3/3: Deploying to Static Web App" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Resource Group: $ResourceGroup" -ForegroundColor Gray
Write-Host "   Static Web App: $StaticWebAppName" -ForegroundColor Gray
Write-Host ""

# Check if swa CLI is available, install if not
$swaCmd = Get-Command swa -ErrorAction SilentlyContinue
if (-not $swaCmd) {
    Write-Host "[INFO] Installing SWA CLI..." -ForegroundColor Yellow
    npm install -g @azure/static-web-apps-cli
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install SWA CLI" -ForegroundColor Red
        exit 1
    }
}

try {
    swa deploy "$ScriptDir\dist" `
        --deployment-token $deploymentToken `
        --env production

    if ($LASTEXITCODE -ne 0) {
        throw "SWA deployment failed"
    }

    Write-Host ""
    Write-Host "[OK] Deployment completed successfully!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Deployment failed: $_" -ForegroundColor Red
    exit 1
}

# Health check
Write-Host ""
Write-Host "[Performing health check]" -ForegroundColor Cyan
Start-Sleep -Seconds 10
try {
    $response = Invoke-WebRequest -Uri "https://$swaHostname" -Method GET -UseBasicParsing -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] Health check PASSED - Frontend is responding" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Health check returned status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARN] Health check failed - Frontend may still be propagating" -ForegroundColor Yellow
    Write-Host "   Error: $_" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "FRONTEND DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Frontend URL: https://$swaHostname" -ForegroundColor Cyan
Write-Host ""
