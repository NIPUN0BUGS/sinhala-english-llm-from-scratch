# Hourly local training and test runner for Windows PowerShell.
# Run setup_hourly_training.ps1 once as the current Windows user.
$ErrorActionPreference = "Continue"
$Repo = "D:\llm\my-llm-git"
$Python = "D:\llm\my-llm\.venv\Scripts\python.exe"
$LogDir = Join-Path $Repo "training-logs"
$Checkpoint = Join-Path $Repo "checkpoints\hourly-model.pt"
$TrainCorpus = Join-Path $Repo "data\processed\train.txt"
$ValCorpus = Join-Path $Repo "data\processed\validation.txt"
$TestCorpus = Join-Path $Repo "data\processed\test.txt"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log([string]$Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Add-Content -LiteralPath (Join-Path $LogDir "hourly-runner.log") -Value $line -Encoding UTF8
    Write-Host $line
}
function Invoke-LoggedPython([string[]]$Arguments, [string]$LogPath) {
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    return $LASTEXITCODE
}

if (-not (Test-Path $Python)) { throw "Python executable not found: $Python" }
if (-not (Test-Path (Join-Path $Repo ".git"))) { throw "Repository not found: $Repo" }

while ($true) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    Write-Log "Starting hourly cycle $stamp"
    $testLog = Join-Path $LogDir "tests-$stamp.log"
    $trainLog = Join-Path $LogDir "train-$stamp.log"
    $testExit = Invoke-LoggedPython @("-m","unittest","discover","-s","tests","-v") $testLog

    if ($testExit -eq 0) {
        Write-Log "Tests passed. Checking processed corpus splits."
        if ((Test-Path $TrainCorpus) -and (Test-Path $ValCorpus) -and (Test-Path $TestCorpus)) {
            $trainArgs = @(
                "-m","src.train",
                "--train-corpus",$TrainCorpus,
                "--validation-corpus",$ValCorpus,
                "--test-corpus",$TestCorpus,
                "--checkpoint",$Checkpoint,
                "--epochs","1",
                "--batch-size","8",
                "--context-length","64"
            )
            if (Test-Path $Checkpoint) {
                $trainArgs += @("--resume", $Checkpoint)
            }
            $trainExit = Invoke-LoggedPython $trainArgs $trainLog
            if ($trainExit -eq 0) {
                Write-Log "Training completed successfully. Checkpoint: $Checkpoint"
            } else {
                Write-Log "Training failed (exit $trainExit). See $trainLog. Next hourly cycle will retry."
            }
        } else {
            Write-Log "Processed splits missing. Preparing corpus from data/raw/train_from_dev.txt."
            $prepLog = Join-Path $LogDir "prepare-$stamp.log"
            $prepExit = Invoke-LoggedPython @(
                "-m","src.prepare_corpus",
                "--input",(Join-Path $Repo "data\raw\train_from_dev.txt"),
                "--output-dir",(Join-Path $Repo "data\processed"),
                "--seed","42"
            ) $prepLog
            if ($prepExit -ne 0) { Write-Log "Corpus preparation failed (exit $prepExit). See $prepLog." }
            else { Write-Log "Corpus prepared. Training will start on the next hourly cycle after tests pass." }
        }
    } else {
        Write-Log "TESTS FAILED (exit $testExit). No training this cycle. Full error: $testLog"
        Write-Log "Automatic code repair is not enabled: this runner does not guess edits to source code. Fixes must be applied and verified before training."
    }

    Write-Log "Cycle complete. Sleeping 1 hour."
    Start-Sleep -Seconds 3600
}
