# Sinhala-English LLM project helper
# Safe by default: does not delete files, overwrite source data, or pull Git changes automatically.

$ErrorActionPreference = "Stop"
$Project = "D:\llm\my-llm-git"
$Python = "D:\llm\my-llm\.venv\Scripts\python.exe"
$Zip = Join-Path $env:USERPROFILE "Downloads\flores_sinhala_english_files.zip"
$Raw = Join-Path $Project "data\raw"

function Show-Section([string]$Title) {
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
}

try {
    if (-not (Test-Path $Project)) {
        throw "Project folder not found: $Project"
    }
    if (-not (Test-Path $Python)) {
        throw "Python environment not found: $Python"
    }

    Set-Location $Project
    New-Item -ItemType Directory -Force -Path $Raw | Out-Null

    Show-Section "Project"
    Write-Host "Folder: $Project"
    & $Python --version

    Show-Section "Optional dataset ZIP"
    if (Test-Path $Zip) {
        $Dev = Join-Path $Raw "dev-00000-of-00001.parquet"
        $DevTest = Join-Path $Raw "devtest-00000-of-00001.parquet"
        if ((-not (Test-Path $Dev)) -or (-not (Test-Path $DevTest))) {
            Write-Host "Found ZIP. Extracting dataset files into the project (existing files may be replaced by same-name files)." -ForegroundColor Yellow
            Expand-Archive -LiteralPath $Zip -DestinationPath $Project -Force
        } else {
            Write-Host "Both Parquet files already exist; leaving them unchanged."
        }
    } else {
        Write-Host "ZIP not found in Downloads; skipping extraction."
        Write-Host "Expected: $Zip"
    }

    Show-Section "Dataset file check"
    foreach ($Name in @("dev-00000-of-00001.parquet", "devtest-00000-of-00001.parquet")) {
        $Path = Join-Path $Raw $Name
        if (Test-Path $Path) {
            $Item = Get-Item $Path
            Write-Host ("FOUND  {0} ({1:N0} bytes)" -f $Name, $Item.Length) -ForegroundColor Green
        } else {
            Write-Host "MISSING $Name" -ForegroundColor Yellow
        }
    }

    Show-Section "Git status and update check"
    $GitAvailable = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
    if ($GitAvailable -and (Test-Path (Join-Path $Project ".git"))) {
        git status --short --branch
        git fetch origin
        if ($LASTEXITCODE -eq 0) {
            $Branch = (git branch --show-current).Trim()
            if ($Branch) {
                $Counts = git rev-list --left-right --count "HEAD...origin/$Branch" 2>$null
                if ($LASTEXITCODE -eq 0 -and $Counts) {
                    $Parts = $Counts -split '\s+'
                    Write-Host "Local-only commits: $($Parts[0]); remote-only commits: $($Parts[1])"
                    if ([int]$Parts[1] -gt 0) {
                        Write-Host "Remote updates exist. Not pulling automatically; inspect changes before syncing." -ForegroundColor Yellow
                    } else {
                        Write-Host "No remote commits waiting for this branch."
                    }
                }
            }
        } else {
            Write-Host "Git fetch failed; continuing with local checks." -ForegroundColor Yellow
        }
    } else {
        Write-Host "This project folder is not a Git clone; skipping Git update check." -ForegroundColor Yellow
    }

    Show-Section "Unit tests"
    & $Python -m unittest discover -s tests -v
    $TestsExit = $LASTEXITCODE
    if ($TestsExit -eq 0) {
        Write-Host "Tests passed." -ForegroundColor Green
    } else {
        Write-Host "Tests failed (exit code $TestsExit). Review output above." -ForegroundColor Red
    }

    Show-Section "Parquet reader check"
    & $Python -c "import pandas, pyarrow; print('pandas:', pandas.__version__); print('pyarrow:', pyarrow.__version__)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pandas/pyarrow are not both installed in the selected Python environment." -ForegroundColor Yellow
        $Answer = Read-Host "Install pandas and pyarrow now? This downloads Python packages (Y/N)"
        if ($Answer -match '^(Y|YES)$') {
            & $Python -m pip install pandas pyarrow
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Parquet packages installed."
                & $Python -c "import pandas as pd; p='data/raw/dev-00000-of-00001.parquet'; import os; print('dev file present:', os.path.exists(p)); print('If present, run the preview command after confirming the file path.')"
            } else {
                Write-Host "Package installation failed. See output above." -ForegroundColor Red
            }
        } else {
            Write-Host "Skipped package installation."
        }
    } else {
        Write-Host "Parquet reader packages are installed."
        if (Test-Path (Join-Path $Raw "dev-00000-of-00001.parquet")) {
            & $Python -c "import pandas as pd; df=pd.read_parquet('data/raw/dev-00000-of-00001.parquet'); print('Columns:', df.columns.tolist()); print('Rows:', len(df)); print(df.head(3).to_string(index=False))"
        }
    }

    Show-Section "Finished"
    Write-Host "No training was started automatically. FLORES dev/devtest files are primarily for evaluation, not general chatbot training."
    Write-Host "Press Enter to close."
    [void](Read-Host)
}
catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Press Enter to close."
    [void](Read-Host)
}
