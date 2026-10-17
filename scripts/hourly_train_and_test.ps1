# Hourly local data-growth, training, and test runner for Windows PowerShell.
# Runs while this process is alive; the scheduled task should launch it at logon.
$ErrorActionPreference = "Continue"
$Repo = "D:\llm\my-llm-git"
$Python = "D:\llm\my-llm\.venv\Scripts\python.exe"
$LogDir = Join-Path $Repo "training-logs"
$Checkpoint = Join-Path $Repo "checkpoints\hourly-model.pt"
$BaseCorpus = Join-Path $Repo "data\raw\train_from_dev.txt"
$CombinedCorpus = Join-Path $Repo "data\raw\hourly_corpus.txt"
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
if (-not (Test-Path $BaseCorpus)) { throw "Seed corpus not found: $BaseCorpus" }

while ($true) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    Write-Log "Starting hourly cycle $stamp"

    # Add a few unique, openly licensed Wikimedia text extracts.
    # Wikipedia text is CC BY-SA; URLs and license metadata are retained in the manifest.
    $dataLog = Join-Path $LogDir "data-update-$stamp.log"
    $dataExit = Invoke-LoggedPython @("-m","src.update_wikipedia_corpus","--pages-per-language","2") $dataLog
    if ($dataExit -eq 0) {
        Write-Log "Dataset refresh completed. New page counts and source attribution are in $dataLog and data\raw\wikipedia_manifest.jsonl."
    } else {
        Write-Log "Dataset refresh returned exit $dataExit. Existing seed/external data will be used if available."
    }

    # Rebuild splits from the combined corpus each cycle so newly collected text is eligible for training.
    $prepLog = Join-Path $LogDir "prepare-$stamp.log"
    $prepExit = Invoke-LoggedPython @(
        "-m","src.prepare_corpus",
        "--input",$CombinedCorpus,
        "--output-dir",(Join-Path $Repo "data\processed"),
        "--seed","42"
    ) $prepLog
    if ($prepExit -ne 0) {
        Write-Log "Corpus preparation failed (exit $prepExit). Training skipped this cycle."
    } else {
        Write-Log "Processed train/validation/test splits refreshed from current corpus."
    }

    $testLog = Join-Path $LogDir "tests-$stamp.log"
    $trainLog = Join-Path $LogDir "train-$stamp.log"
    $testExit = Invoke-LoggedPython @("-m","unittest","discover","-s","tests","-v") $testLog

    if ($prepExit -eq 0 -and $testExit -eq 0) {
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
                Write-Log "Training completed. Checkpoint: $Checkpoint"
            } else {
                Write-Log "Training failed (exit $trainExit). See $trainLog. Next hourly cycle will retry."
            }
        } else {
            Write-Log "Processed splits are missing despite successful preparation; training skipped."
        }
    } else {
        if ($prepExit -ne 0) { Write-Log "TRAINING SKIPPED: corpus preparation failed; see $prepLog." }
        if ($testExit -ne 0) {
            Write-Log "TESTS FAILED (exit $testExit). Full error: $testLog"
            Write-Log "Automatic source-code repair is not enabled; this runner adds data and retries tests next hour but does not guess code edits."
        }
    }

    Write-Log "Cycle complete. Sleeping 1 hour."
    Start-Sleep -Seconds 3600
}
