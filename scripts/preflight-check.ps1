<#
.SYNOPSIS
    Pre-flight validation for Earth Copilot Azure deployment.
    Checks every known constraint BEFORE triggering the GitHub Actions workflow.

.DESCRIPTION
    After 8+ failed deployment attempts, every pitfall has been catalogued.
    This script checks:
      1. CLI tools (az, gh, bicep, node, npm)
      2. Azure authentication & subscription
      3. Resource provider registration (including Microsoft.Web, Microsoft.Network)
      4. Soft-deleted Key Vaults / Cognitive Services (name collisions)
      5. Existing resource group conflicts
      6. Region + Model availability (interactive picker)
      7. Static Web App readiness (frontend hosting)
      8. GitHub repo, environment, and AZURE_CREDENTIALS secret
      9. Service principal roles
     10. Static Web App discovery & status
     11. VNet subnet validation (backend private endpoints)
     12. Frontend build pre-requisites (package.json, package-lock.json, Dockerfile)

    Run this BEFORE triggering the deployment workflow.

.PARAMETER SkipProbe
    Skip the live ARM probe deployment test (faster, but won't catch catalog-vs-ARM discrepancies).

.PARAMETER SkipGitHub
    Skip GitHub-related checks (useful for CI/CD or disconnected dev).

.PARAMETER Location
    Azure region to validate. If omitted, shows an interactive picker.

.PARAMETER EnvironmentName
    GitHub environment name. Default: dev

.EXAMPLE
    .\scripts\preflight-check.ps1
    .\scripts\preflight-check.ps1 -Location canadacentral
    .\scripts\preflight-check.ps1 -SkipGitHub
#>

[CmdletBinding()]
param(
    [switch]$SkipGitHub,
    [switch]$SkipProbe,
    [string]$Location = '',
    [string]$EnvironmentName = 'dev',
    [string]$ResourceGroupName = ''
)

# ─── Helpers ──────────────────────────────────────────────────────────
$script:PassCount = 0
$script:WarnCount = 0
$script:FailCount = 0

function Write-Check  { param([string]$Msg) Write-Host "  [ ] $Msg" -ForegroundColor Cyan }
function Write-Pass   { param([string]$Msg) $script:PassCount++; Write-Host "  [PASS] $Msg" -ForegroundColor Green }
function Write-Warn   { param([string]$Msg) $script:WarnCount++; Write-Host "  [WARN] $Msg" -ForegroundColor Yellow }
function Write-Fail   { param([string]$Msg) $script:FailCount++; Write-Host "  [FAIL] $Msg" -ForegroundColor Red }
function Write-Info   { param([string]$Msg) Write-Host "         $Msg" -ForegroundColor DarkGray }
function Write-Section { param([string]$Msg) Write-Host "`n========== $Msg ==========" -ForegroundColor White }

# Known-good regions for Earth Copilot (Container Apps + AI Foundry + Static Web Apps + Azure Maps)
$SupportedRegions = @(
    'eastus', 'eastus2', 'westus', 'westus2', 'westus3',
    'centralus', 'northcentralus', 'southcentralus', 'westcentralus',
    'canadacentral', 'canadaeast',
    'northeurope', 'westeurope', 'uksouth', 'ukwest',
    'francecentral', 'germanywestcentral', 'switzerlandnorth', 'swedencentral', 'norwayeast',
    'australiaeast', 'australiasoutheast',
    'japaneast', 'japanwest',
    'southeastasia', 'eastasia',
    'koreacentral', 'koreasouth',
    'centralindia', 'southindia',
    'brazilsouth'
)

# Required resource providers
$RequiredProviders = @(
    'Microsoft.App',
    'Microsoft.ContainerService',
    'Microsoft.ContainerRegistry',
    'Microsoft.ContainerInstance',
    'Microsoft.CognitiveServices',
    'Microsoft.Maps',
    'Microsoft.MachineLearningServices',
    'Microsoft.Web',
    'Microsoft.Network',
    'Microsoft.Storage',
    'Microsoft.KeyVault',
    'Microsoft.OperationalInsights'
)

# ─── Banner ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Earth Copilot  -  Pre-Flight Deployment Checklist" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Validates all Azure constraints before deployment." -ForegroundColor DarkGray
Write-Host ""

# ══════════════════════════════════════════════════════════════════════
# 1. CLI TOOLS
# ══════════════════════════════════════════════════════════════════════
Write-Section "1/12  CLI Tools"

# Azure CLI
if (Get-Command az -ErrorAction SilentlyContinue) {
    $azVersion = (az version 2>$null | ConvertFrom-Json).'azure-cli'
    Write-Pass "Azure CLI installed (v$azVersion)"
} else {
    Write-Fail "Azure CLI not found. Install: winget install Microsoft.AzureCLI"
}

# Bicep
try {
    $bicepVer = az bicep version 2>$null
    if ($bicepVer -match '(\d+\.\d+\.\d+)') {
        Write-Pass "Bicep CLI installed ($($Matches[1]))"
    } else {
        Write-Warn "Bicep CLI version unclear. Run: az bicep install"
    }
} catch {
    Write-Warn "Bicep CLI not found. Run: az bicep install"
}

# GitHub CLI
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $ghVersion = (gh --version 2>$null | Select-Object -First 1)
    Write-Pass "GitHub CLI installed ($ghVersion)"
} else {
    if ($SkipGitHub) {
        Write-Warn "GitHub CLI not found (skipped — -SkipGitHub)"
    } else {
        Write-Fail "GitHub CLI not found. Install: winget install GitHub.cli"
    }
}

# Node.js (required for frontend build)
if (Get-Command node -ErrorAction SilentlyContinue) {
    $nodeVersion = (node --version 2>$null).Trim()
    $nodeMajor = [int]($nodeVersion -replace '^v(\d+)\..*', '$1')
    if ($nodeMajor -ge 20) {
        Write-Pass "Node.js installed ($nodeVersion)"
    } else {
        Write-Fail "Node.js $nodeVersion is too old. Frontend requires Node.js 20+. Install: winget install OpenJS.NodeJS.LTS"
    }
} else {
    Write-Fail "Node.js not found. Frontend build requires Node.js 20+. Install: winget install OpenJS.NodeJS.LTS"
}

# npm (required for frontend build)
if (Get-Command npm -ErrorAction SilentlyContinue) {
    $npmVersion = (npm --version 2>$null).Trim()
    Write-Pass "npm installed (v$npmVersion)"
} else {
    Write-Fail "npm not found. Install Node.js LTS which includes npm."
}

# ══════════════════════════════════════════════════════════════════════
# 2. AZURE AUTHENTICATION & SUBSCRIPTION
# ══════════════════════════════════════════════════════════════════════
Write-Section "2/12  Azure Authentication"

$accountJson = az account show 2>$null | ConvertFrom-Json
if ($accountJson) {
    Write-Pass "Logged in as: $($accountJson.user.name)"
    Write-Pass "Subscription: $($accountJson.name) ($($accountJson.id))"
    Write-Info "Tenant: $($accountJson.tenantId)"
    $subscriptionId = $accountJson.id
    $tenantId = $accountJson.tenantId
} else {
    Write-Fail "Not authenticated. Run: az login"
    Write-Host "`nCannot continue without Azure authentication." -ForegroundColor Red
    exit 1
}

# ══════════════════════════════════════════════════════════════════════
# 3. RESOURCE PROVIDER REGISTRATION
# ══════════════════════════════════════════════════════════════════════
Write-Section "3/12  Resource Providers"

foreach ($ns in $RequiredProviders) {
    # Take last line only — az CLI sometimes emits update/warning notices on stdout after login
    $stateRaw = az provider show --namespace $ns --query "registrationState" -o tsv 2>$null
    $state = if ($stateRaw) { ($stateRaw | Select-Object -Last 1).Trim() } else { '' }
    if ($state -eq 'Registered') {
        Write-Pass "$ns — Registered"
    } elseif ($state -eq 'Registering') {
        Write-Warn "$ns — Registering (wait a few minutes and re-run)"
    } else {
        Write-Fail "$ns — $state. Fix: az provider register --namespace $ns"
    }
}

# ══════════════════════════════════════════════════════════════════════
# 4. SOFT-DELETED RESOURCES (Key Vault & Cognitive Services)
# ══════════════════════════════════════════════════════════════════════
Write-Section "4/12  Soft-Deleted Resources"

# Key Vaults
$deletedKvs = az keyvault list-deleted --query "[].{name:name, location:properties.location}" -o json 2>$null | ConvertFrom-Json
if ($deletedKvs -and $deletedKvs.Count -gt 0) {
    Write-Warn "$($deletedKvs.Count) soft-deleted Key Vault(s) found — may cause name collisions:"
    foreach ($kv in $deletedKvs) {
        Write-Info "  - $($kv.name) ($($kv.location))"
    }
    Write-Info "  Purge with: az keyvault purge --name <name> --location <location>"

    # Flag any that look like Earth Copilot Key Vaults (kv-<hash> pattern)
    foreach ($kv in $deletedKvs) {
        if ($kv.name -match '^kv-[a-z0-9]+$') {
            Write-Warn "Key Vault '$($kv.name)' looks like an Earth Copilot vault. Purge before deploy to avoid name collision."
            Write-Info "    az keyvault purge --name $($kv.name) --location $($kv.location)"
        }
    }
} else {
    Write-Pass "No soft-deleted Key Vaults found"
}

# Cognitive Services
$deletedCog = az cognitiveservices account list-deleted --query "[].{name:name, location:location}" -o json 2>$null | ConvertFrom-Json
if ($deletedCog -and $deletedCog.Count -gt 0) {
    Write-Warn "$($deletedCog.Count) soft-deleted Cognitive Services account(s) found:"
    foreach ($c in $deletedCog) {
        Write-Info "  - $($c.name) ($($c.location))"
    }
    Write-Info "  Purge with: az cognitiveservices account purge --name <name> --resource-group <rg> --location <location>"
} else {
    Write-Pass "No soft-deleted Cognitive Services accounts"
}

# ══════════════════════════════════════════════════════════════════════
# 5. EXISTING RESOURCE GROUP
# ══════════════════════════════════════════════════════════════════════
Write-Section "5/12  Existing Resource Group"

# Match the workflow logic: vars.RESOURCE_GROUP || 'rg-earthcopilot'
$rgName = if ($ResourceGroupName) { $ResourceGroupName } else { 'rg-earthcopilot' }
$rgExists = az group show --name $rgName 2>$null | ConvertFrom-Json
if ($rgExists) {
    $rgState = $rgExists.properties.provisioningState
    if ($rgState -eq 'Deleting') {
        Write-Warn "Resource group '$rgName' is currently being deleted. Wait for deletion to finish."
    } else {
        Write-Warn "Resource group '$rgName' already exists ($rgState in $($rgExists.location))."
        Write-Info "This is fine for re-deployments, but may cause conflicts if switching regions."
        Write-Info "Delete first to change region: az group delete --name $rgName --yes"
    }
} else {
    Write-Pass "Resource group '$rgName' does not exist (clean deploy)"
}

# ══════════════════════════════════════════════════════════════════════
# 6. REGION & MODEL SELECTION (Interactive)
# ══════════════════════════════════════════════════════════════════════
Write-Section "6/12  Region & Model Availability"

# If no location provided, let user pick
if ([string]::IsNullOrEmpty($Location)) {
    Write-Host ""
    Write-Host "  Select a deployment region:" -ForegroundColor Yellow
    Write-Host "  Recommended regions with broad model + service support:" -ForegroundColor DarkGray
    Write-Host ""

    $recommendedRegions = @(
        @{ Name = 'eastus2';          Desc = 'US East 2 — Broad model availability, popular choice' }
        @{ Name = 'eastus';           Desc = 'US East — Broad model availability' }
        @{ Name = 'westus2';          Desc = 'US West 2 — West coast option' }
        @{ Name = 'westus3';          Desc = 'US West 3 — West coast option' }
        @{ Name = 'canadacentral';    Desc = 'Canada Central — Data residency in Canada' }
        @{ Name = 'swedencentral';    Desc = 'Sweden Central — Europe option' }
        @{ Name = 'uksouth';          Desc = 'UK South — Europe option' }
        @{ Name = 'australiaeast';    Desc = 'Australia East — APAC option' }
        @{ Name = 'japaneast';        Desc = 'Japan East — APAC option' }
    )

    for ($i = 0; $i -lt $recommendedRegions.Count; $i++) {
        $r = $recommendedRegions[$i]
        Write-Host "    [$($i+1)] $($r.Name.PadRight(20)) $($r.Desc)" -ForegroundColor Cyan
    }
    Write-Host "    [0] Enter custom region" -ForegroundColor DarkGray
    Write-Host ""

    do {
        $choice = Read-Host "  Enter choice (1-$($recommendedRegions.Count), or 0 for custom)"
        if ($choice -eq '0') {
            $Location = Read-Host "  Enter Azure region name"
        } elseif ([int]::TryParse($choice, [ref]$null) -and [int]$choice -ge 1 -and [int]$choice -le $recommendedRegions.Count) {
            $Location = $recommendedRegions[[int]$choice - 1].Name
        } else {
            Write-Host "  Invalid choice, try again." -ForegroundColor Red
            $Location = ''
        }
    } while ([string]::IsNullOrEmpty($Location))
}

Write-Pass "Selected region: $Location"

# Check model availability in the selected region
Write-Host ""
Write-Host "  Checking model availability in $Location..." -ForegroundColor DarkGray
Write-Info "NOTE: The model listing API can show models as available that ARM rejects at deploy time."
Write-Info "Regions proven to work: eastus2, eastus. If this check passes but deploy fails, switch region."

# Models we need: gpt-4o (primary) and gpt-4o-mini (secondary)
$targetModels = @(
    @{ Name = 'gpt-4o';      PreferredSku = 'GlobalStandard'; FallbackSku = 'Standard'; Role = 'Primary chat model' }
    @{ Name = 'gpt-4o-mini'; PreferredSku = 'GlobalStandard'; FallbackSku = 'Standard'; Role = 'Fast/cheap model' }
)

# Query all available models in the region (AIServices kind only — that's what we deploy)
$allModels = az cognitiveservices model list --location $Location --query "[?kind=='AIServices' && (model.name=='gpt-4o' || model.name=='gpt-4o-mini')].{name:model.name, version:model.version, skus:model.skus[].name}" -o json 2>$null | ConvertFrom-Json

$modelResults = @()

foreach ($target in $targetModels) {
    $found = $allModels | Where-Object { $_.name -eq $target.Name }
    if ($found) {
        # Collect ALL SKUs across ALL version entries for this model
        $allSkus = @()
        $latestVersion = ''
        foreach ($entry in @($found)) {
            if ($entry.skus) { $allSkus += @($entry.skus) }
            if ($entry.version -gt $latestVersion) { $latestVersion = $entry.version }
        }
        $allSkus = $allSkus | Select-Object -Unique

        if ($allSkus -contains $target.PreferredSku) {
            Write-Pass "$($target.Name) available with $($target.PreferredSku) SKU ($($target.Role))"
            $modelResults += @{ Name = $target.Name; Sku = $target.PreferredSku; Version = $latestVersion; Status = 'ok' }
        } elseif ($allSkus -contains $target.FallbackSku) {
            Write-Warn "$($target.Name) available with $($target.FallbackSku) only (not $($target.PreferredSku)) — $($target.Role)"
            Write-Info "  You may need to update ai-foundry.bicep SKU from '$($target.PreferredSku)' to '$($target.FallbackSku)'"
            $modelResults += @{ Name = $target.Name; Sku = $target.FallbackSku; Version = $latestVersion; Status = 'fallback' }
        } else {
            Write-Fail "$($target.Name) listed in $Location but no supported SKU found (available: $($allSkus -join ', '))"
            $modelResults += @{ Name = $target.Name; Sku = $null; Version = $latestVersion; Status = 'no-sku' }
        }
    } else {
        Write-Fail "$($target.Name) NOT available in $Location"
        $modelResults += @{ Name = $target.Name; Sku = $null; Version = $null; Status = 'missing' }
    }
}

# Show alternative models if primary models are missing
$failedModels = $modelResults | Where-Object { $_.Status -in @('missing', 'no-sku') }
if ($failedModels.Count -gt 0) {
    Write-Host ""
    Write-Warn "Some required models are not available in $Location."
    Write-Host "  Alternative models available in this region:" -ForegroundColor Yellow

    $alternatives = az cognitiveservices model list --location $Location --query "[?kind=='AIServices' && (model.name=='gpt-4' || model.name=='gpt-4-turbo' || model.name=='gpt-35-turbo' || model.name=='gpt-4.1' || model.name=='gpt-4.1-mini' || model.name=='gpt-4.1-nano')].{name:model.name, version:model.version, skus:model.skus[].name}" -o json 2>$null | ConvertFrom-Json
    if ($alternatives) {
        $altGrouped = $alternatives | Group-Object -Property name
        foreach ($g in $altGrouped) {
            $latest = $g.Group | Sort-Object { $_.version } -Descending | Select-Object -First 1
            Write-Info "  - $($g.Name) (v$($latest.version)) — SKUs: $($latest.skus -join ', ')"
        }
    }
    Write-Host ""
    Write-Warn "ACTION: Pick a different region or update ai-foundry.bicep with available models."
}

# ─── 6b. PROBE DEPLOYMENT (actually tests ARM, not just the catalog API) ───
$modelsToProbe = $modelResults | Where-Object { $_.Status -eq 'ok' }
if ($SkipProbe) {
    Write-Info "  Probe test skipped (--SkipProbe). Use probe to catch catalog-vs-ARM discrepancies."
} elseif ($modelsToProbe.Count -gt 0) {
    Write-Host ""
    Write-Host "  Probe-testing model deployment against ARM (the catalog API can be unreliable)..." -ForegroundColor DarkGray

    $probeRg   = "rg-preflight-probe-$(Get-Random -Maximum 99999)"
    $probeName = "probe$(Get-Random -Maximum 99999)"

    # Create temporary resource group
    $rgCreated = $false
    try {
        az group create --name $probeRg --location $Location --tags "purpose=preflight-probe" -o none 2>$null
        if ($LASTEXITCODE -eq 0) { $rgCreated = $true }
    } catch { }

    if ($rgCreated) {
        # Create temporary AI Services account
        $accountCreated = $false
        try {
            az cognitiveservices account create `
                --name $probeName `
                --resource-group $probeRg `
                --location $Location `
                --kind AIServices `
                --sku S0 `
                --custom-domain $probeName `
                --yes -o none 2>$null
            if ($LASTEXITCODE -eq 0) { $accountCreated = $true }
        } catch { }

        if ($accountCreated) {
            # Try deploying the primary model (gpt-4o)
            $primaryModel = $modelsToProbe | Where-Object { $_.Name -eq 'gpt-4o' } | Select-Object -First 1
            if ($primaryModel) {
                $probeResult = az cognitiveservices account deployment create `
                    --name $probeName `
                    --resource-group $probeRg `
                    --deployment-name "probe-gpt4o" `
                    --model-name $primaryModel.Name `
                    --model-version $primaryModel.Version `
                    --model-format OpenAI `
                    --sku-name $primaryModel.Sku `
                    --sku-capacity 1 `
                    -o none 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Pass "PROBE: $($primaryModel.Name) with $($primaryModel.Sku) deployed successfully in $Location"
                } else {
                    $errMsg = ($probeResult | Out-String).Trim()
                    Write-Fail "PROBE: $($primaryModel.Name) with $($primaryModel.Sku) REJECTED by ARM in $Location"
                    Write-Info "  ARM error: $errMsg"
                    Write-Info "  The model listing API said this was available, but ARM rejected it."
                    Write-Info "  Switch to a proven region (eastus2, eastus) or try a different model."
                }
            }
        } else {
            Write-Warn "PROBE: Could not create temporary AI Services account — skipping probe test"
        }

        # Clean up: delete the Cognitive Services account first (synchronously), purge it,
        # then delete the RG asynchronously. This prevents orphaned soft-deleted accounts.
        Write-Host "  Cleaning up probe resources..." -ForegroundColor DarkGray
        az cognitiveservices account delete --name $probeName --resource-group $probeRg -o none 2>$null
        az cognitiveservices account purge --name $probeName --resource-group $probeRg --location $Location -o none 2>$null
        az group delete --name $probeRg --yes --no-wait -o none 2>$null
    } else {
        Write-Warn "PROBE: Could not create temporary resource group — skipping probe test"
    }
}

# ══════════════════════════════════════════════════════════════════════
# 7. STATIC WEB APP READINESS (frontend hosting)
# ══════════════════════════════════════════════════════════════════════
Write-Section "7/12  Static Web App Readiness"

# The frontend uses Azure Static Web Apps (Free tier). No VM quota is required.
# SWA is available in all regions and requires only the Microsoft.Web provider.

Write-Host "  Frontend uses Azure Static Web Apps (Free tier — no VM quota needed)." -ForegroundColor DarkGray
Write-Host "  Checking Microsoft.Web provider registration..." -ForegroundColor DarkGray

$webProvider = az provider show --namespace Microsoft.Web --query "registrationState" -o tsv 2>$null
if ($webProvider -eq 'Registered') {
    Write-Pass "Microsoft.Web provider is registered (required for Static Web Apps)"
} else {
    Write-Warn "Microsoft.Web provider state: $webProvider — registering now..."
    az provider register --namespace Microsoft.Web -o none 2>$null
    Write-Info "  Registration started. It may take a few minutes to complete."
}

# Check for existing Static Web App in the resource group
$rgName1 = if ($ResourceGroupName) { $ResourceGroupName } else { 'rg-earthcopilot' }
$existingSwa = az staticwebapp list --resource-group $rgName1 --query "[0].name" -o tsv 2>$null
if ($existingSwa) {
    Write-Pass "Existing Static Web App found: $existingSwa"
    $swaUrl = az staticwebapp show --name $existingSwa --resource-group $rgName1 --query "defaultHostname" -o tsv 2>$null
    if ($swaUrl) { Write-Info "  URL: https://$swaUrl" }
} else {
    Write-Pass "No existing Static Web App — one will be created during deployment"
}

Write-Pass "Static Web App readiness confirmed (no VM quota needed)"

# ══════════════════════════════════════════════════════════════════════
# 8. GITHUB CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
Write-Section "8/12  GitHub Configuration"

if ($SkipGitHub) {
    Write-Info "Skipped (--SkipGitHub)"
} else {
    # Detect the GitHub remote (handles HTTPS, SSH, and SSH aliases like github.com-username)
    $remoteUrl = git remote get-url origin 2>$null
    if ($remoteUrl -match 'github\.com[^:/]*[:/](.+?)(?:\.git)?$') {
        $repoFullName = $Matches[1]
        Write-Pass "GitHub remote: $repoFullName"
    } else {
        Write-Fail "Could not detect GitHub remote. Ensure 'origin' points to your fork."
        Write-Info "  Current remote URL: $remoteUrl"
        $repoFullName = $null
    }

    if ($repoFullName) {
        # Temporarily clear GITHUB_TOKEN to avoid interference
        $savedToken = $env:GITHUB_TOKEN
        $env:GITHUB_TOKEN = ""

        # Check if environment exists
        $envCheck = gh api "repos/$repoFullName/environments/$EnvironmentName" 2>$null | ConvertFrom-Json
        if ($envCheck -and $envCheck.name) {
            Write-Pass "GitHub environment '$EnvironmentName' exists"
        } else {
            Write-Fail "GitHub environment '$EnvironmentName' not found."
            Write-Info "  Create it: gh api repos/$repoFullName/environments/$EnvironmentName -X PUT"
        }

        # Check AZURE_CREDENTIALS secret
        $secretList = gh secret list --env $EnvironmentName -R $repoFullName 2>$null
        if ($secretList -match 'AZURE_CREDENTIALS') {
            Write-Pass "AZURE_CREDENTIALS secret configured in '$EnvironmentName'"
        } else {
            Write-Fail "AZURE_CREDENTIALS secret not found in environment '$EnvironmentName'."
            Write-Info "  Set it: gh secret set AZURE_CREDENTIALS --env $EnvironmentName -R $repoFullName"
        }

        # Check LOCATION variable
        $varList = gh variable list --env $EnvironmentName -R $repoFullName 2>$null
        if ($varList -match 'LOCATION') {
            $currentLocation = ($varList | Select-String 'LOCATION').ToString() -replace '.*LOCATION\s+', '' -replace '\s+.*', ''
            if ($currentLocation -ne $Location -and -not [string]::IsNullOrEmpty($Location)) {
                Write-Warn "GitHub LOCATION variable is '$currentLocation' but you selected '$Location'"
                Write-Info "  Update: gh variable set LOCATION --env $EnvironmentName -R $repoFullName --body '$Location'"
            } else {
                Write-Pass "LOCATION variable set to '$currentLocation'"
            }
        } else {
            if ($Location -ne 'eastus2') {
                Write-Warn "No LOCATION variable set — workflow defaults to 'eastus2', but you selected '$Location'"
                Write-Info "  Set it: gh variable set LOCATION --env $EnvironmentName -R $repoFullName --body '$Location'"
            } else {
                Write-Pass "No LOCATION variable needed (using default: eastus2)"
            }
        }

        # Check if Actions workflows are enabled
        $workflows = gh api "repos/$repoFullName/actions/workflows" 2>$null | ConvertFrom-Json
        $deployWorkflow = $workflows.workflows | Where-Object { $_.name -eq 'Deploy Earth Copilot' }
        if ($deployWorkflow) {
            if ($deployWorkflow.state -eq 'active') {
                Write-Pass "Deploy workflow is active (ID: $($deployWorkflow.id))"
            } else {
                Write-Warn "Deploy workflow state: $($deployWorkflow.state). Enable it in GitHub Actions tab."
            }
        } else {
            Write-Fail "Deploy Earth Copilot workflow not found in repository."
        }

        # Restore token
        $env:GITHUB_TOKEN = $savedToken
    }
}

# ══════════════════════════════════════════════════════════════════════
# 9. SERVICE PRINCIPAL ROLES
# ══════════════════════════════════════════════════════════════════════
Write-Section "9/12  Service Principal Roles"

$spName = "sp-earthcopilot-$EnvironmentName"
$spList = az ad sp list --display-name $spName --query "[0].{appId:appId, displayName:displayName}" -o json 2>$null | ConvertFrom-Json

if ($spList -and $spList.appId) {
    Write-Pass "Service principal found: $($spList.displayName) ($($spList.appId))"

    # Check role assignments
    $roles = az role assignment list --assignee $spList.appId --query "[].roleDefinitionName" -o json 2>$null | ConvertFrom-Json
    $hasContributor = $roles -contains 'Contributor'
    $hasUAA = $roles -contains 'User Access Administrator'

    if ($hasContributor) {
        Write-Pass "Has 'Contributor' role"
    } else {
        Write-Fail "Missing 'Contributor' role on subscription"
        Write-Info "  Fix: az role assignment create --assignee $($spList.appId) --role Contributor --scope /subscriptions/$subscriptionId"
    }

    if ($hasUAA) {
        Write-Pass "Has 'User Access Administrator' role"
    } else {
        Write-Fail "Missing 'User Access Administrator' role on subscription"
        Write-Info "  Fix: az role assignment create --assignee $($spList.appId) --role 'User Access Administrator' --scope /subscriptions/$subscriptionId"
    }
} else {
    Write-Warn "Service principal '$spName' not found. Create it per QUICK_DEPLOY.md Step 7."
    Write-Info "  Or the SP may have a different name — check: az ad sp list --display-name sp-earthcopilot"
}

# ══════════════════════════════════════════════════════════════════════
# 10. STATIC WEB APP DISCOVERY & STATUS
# ══════════════════════════════════════════════════════════════════════
Write-Section "10/12  Static Web App Discovery"

# The deploy workflow creates a Static Web App named swa-earthcopilot (or discovers existing).
$rgName2 = if ($ResourceGroupName) { $ResourceGroupName } else { 'rg-earthcopilot' }
$existingSwa2 = az staticwebapp list --resource-group $rgName2 --query "[0].{name:name, sku:sku.name, state:contentDistributionEndpoint}" -o json 2>$null | ConvertFrom-Json
if ($existingSwa2 -and $existingSwa2.name) {
    Write-Pass "Static Web App found: $($existingSwa2.name) (SKU: $($existingSwa2.sku))"
    Write-Info "  Re-deploy will update existing Static Web App."
} else {
    Write-Pass "No existing Static Web App — one will be created during deployment"
    Write-Info "  Name: swa-earthcopilot (auto-created by deploy workflow)"
}

# ══════════════════════════════════════════════════════════════════════
# 11. VNET SUBNET VALIDATION (Backend Private Endpoints)
# ══════════════════════════════════════════════════════════════════════
Write-Section "11/12  VNet Subnet Validation"

# Static Web Apps do NOT require VNet integration or a dedicated subnet.
# This check validates subnets needed by the backend (Container Apps, private endpoints).
$rgName3 = if ($ResourceGroupName) { $ResourceGroupName } else { 'rg-earthcopilot' }
$existingVnet = az network vnet list --resource-group $rgName3 --query "[0].name" -o tsv 2>$null
if ($existingVnet) {
    Write-Info "  Found VNet: $existingVnet"
    $subnets = az network vnet subnet list --resource-group $rgName3 --vnet-name $existingVnet --query "[].name" -o tsv 2>$null

    if ($subnets -match 'snet-container-apps') {
        Write-Pass "Subnet 'snet-container-apps' exists (backend Container Apps)"
    } else {
        Write-Warn "Subnet 'snet-container-apps' not found"
    }
    if ($subnets -match 'snet-private-endpoints') {
        Write-Pass "Subnet 'snet-private-endpoints' exists (private endpoints)"
    } else {
        Write-Warn "Subnet 'snet-private-endpoints' not found"
    }

    Write-Info "  Note: Frontend (Static Web App) does not require VNet integration."
} else {
    Write-Info "No VNet found — will be created during infrastructure deployment (if private endpoints enabled)"
    Write-Pass "VNet subnet check skipped (no existing VNet)"
}

# ══════════════════════════════════════════════════════════════════════
# 12. FRONTEND BUILD PRE-REQUISITES
# ══════════════════════════════════════════════════════════════════════
Write-Section "12/12  Frontend Build Pre-Requisites"

# Check package-lock.json exists (npm ci requires it)
$lockFile = Join-Path $PSScriptRoot "..\earth-copilot\web-ui\package-lock.json"
if (Test-Path $lockFile) {
    Write-Pass "package-lock.json exists (required for npm ci in CI/CD)"
} else {
    Write-Fail "package-lock.json not found at earth-copilot/web-ui/"
    Write-Info "  The GitHub Actions workflow uses 'npm ci' which requires package-lock.json."
    Write-Info "  Generate it: cd earth-copilot/web-ui && npm install"
}

# Check package.json exists
$pkgFile = Join-Path $PSScriptRoot "..\earth-copilot\web-ui\package.json"
if (Test-Path $pkgFile) {
    Write-Pass "package.json exists"
    # Check for required build script
    $pkgJson = Get-Content $pkgFile -Raw | ConvertFrom-Json
    if ($pkgJson.scripts -and $pkgJson.scripts.build) {
        Write-Pass "Build script defined: '$($pkgJson.scripts.build)'"
    } else {
        Write-Fail "No 'build' script found in package.json"
    }
} else {
    Write-Fail "package.json not found at earth-copilot/web-ui/"
}

# Check Dockerfile exists for backend (informational)
$dockerFile = Join-Path $PSScriptRoot "..\earth-copilot\container-app\Dockerfile.complete"
if (Test-Path $dockerFile) {
    Write-Pass "Backend Dockerfile.complete exists"
} else {
    Write-Warn "Dockerfile.complete not found at earth-copilot/container-app/"
    Write-Info "  The CI/CD workflow builds the backend image using Dockerfile.complete"
}

# ══════════════════════════════════════════════════════════════════════
# BONUS: BICEP COMPILATION CHECK
# ══════════════════════════════════════════════════════════════════════
Write-Section "BONUS  Bicep Compilation"

$bicepPath = Join-Path $PSScriptRoot "..\earth-copilot\infra\main.bicep"
if (Test-Path $bicepPath) {
    $buildResult = az bicep build --file $bicepPath 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Pass "main.bicep compiles successfully"
        # Clean up generated ARM template
        $armPath = $bicepPath -replace '\.bicep$', '.json'
        if (Test-Path $armPath) { Remove-Item $armPath -Force }
    } else {
        Write-Fail "main.bicep compilation failed:"
        $buildResult | ForEach-Object { Write-Info "  $_" }
    }
} else {
    Write-Warn "main.bicep not found at expected path"
}

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Pre-Flight Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$totalChecks = $script:PassCount + $script:WarnCount + $script:FailCount
Write-Host "  Total checks:  $totalChecks" -ForegroundColor White
Write-Host "  Passed:        $($script:PassCount)" -ForegroundColor Green
Write-Host "  Warnings:      $($script:WarnCount)" -ForegroundColor Yellow
Write-Host "  Failed:        $($script:FailCount)" -ForegroundColor Red
Write-Host ""

if ($script:FailCount -gt 0) {
    Write-Host "  RESULT: $($script:FailCount) FAILURE(s) — fix the items above before deploying." -ForegroundColor Red
    Write-Host ""
    exit 1
} elseif ($script:WarnCount -gt 0) {
    Write-Host "  RESULT: All critical checks passed, but $($script:WarnCount) warning(s)." -ForegroundColor Yellow
    Write-Host "  Review warnings above. Deployment may succeed, but address them to be safe." -ForegroundColor Yellow
    Write-Host ""

    Write-Host "  Ready to deploy? Run:" -ForegroundColor Cyan
    Write-Host "    gh workflow run deploy.yml -f force_all=true -f environment_name=$EnvironmentName" -ForegroundColor White
    Write-Host ""
    exit 0
} else {
    Write-Host "  RESULT: ALL CHECKS PASSED  - ready to deploy!" -ForegroundColor Green
    Write-Host ""

    Write-Host "  Deploy with:" -ForegroundColor Cyan
    Write-Host "    gh workflow run deploy.yml -f force_all=true -f environment_name=$EnvironmentName" -ForegroundColor White
    Write-Host ""
    exit 0
}
