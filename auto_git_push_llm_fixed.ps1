# Auto-commit and push eligible source/test changes for the local LLM repository.
# Safe scope: watches only src/ and tests/; never force-pushes.
# Stop with Ctrl+C.

$ErrorActionPreference = "Stop"
$Repo = "D:\llm\my-llm-git"
$ExpectedBranch = "feature/tiny-transformer"
$Remote = "origin"
$DebounceSeconds = 12
$PollSeconds = 3

# Only these file types are eligible. Secret/data/checkpoint files are explicitly excluded.
$AllowedExtensions = @(".py", ".pyi", ".md", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".txt", ".ps1", ".sh")
$BlockedNames = @(".env", ".env.local", ".env.production", "secrets.json", "credentials.json")
$BlockedExtensions = @(".parquet", ".zip", ".pt", ".pth", ".ckpt", ".safetensors", ".sqlite", ".db", ".pem", ".key", ".pfx")

function Invoke-Git {
    param([Parameter(Mandatory=$true)][string[]]$GitArgs)
    $output = & git -C $Repo @GitArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed:`n$($output -join "`n")"
    }
    return $output
}

function Test-EligiblePath {
    param([string]$RelativePath)
    $p = $RelativePath.Replace('\','/')
    if ($p -notmatch '^(src|tests)/') { return $false }
    $leaf = [IO.Path]::GetFileName($p).ToLowerInvariant()
    if ($BlockedNames -contains $leaf) { return $false }
    $ext = [IO.Path]::GetExtension($leaf)
    if ($BlockedExtensions -contains $ext) { return $false }
    if ($leaf -match '(^|\.)(secret|credential|token|privatekey)(\.|$)') { return $false }
    return $AllowedExtensions -contains $ext
}

function Get-EligibleGitPaths {
    $paths = @()
    $paths += Invoke-Git @("diff", "--name-only", "--", "src", "tests")
    $paths += Invoke-Git @("diff", "--cached", "--name-only", "--", "src", "tests")
    $paths += Invoke-Git @("ls-files", "--others", "--exclude-standard", "--", "src", "tests")
    $unique = @($paths | ForEach-Object { "$_".Trim() } | Where-Object { $_ } | Sort-Object -Unique | Where-Object { Test-EligiblePath $_ })
    return $unique
}

function Get-Snapshot {
    $entries = @()
    foreach ($folder in @("src", "tests")) {
        $full = Join-Path $Repo $folder
        if (Test-Path $full) {
            Get-ChildItem -LiteralPath $full -File -Recurse -Force -ErrorAction SilentlyContinue |
                ForEach-Object {
                    $relative = $_.FullName.Substring($Repo.TrimEnd("\").Length).TrimStart("\", "/")
                    if (Test-EligiblePath $relative) {
                        $entries += "$relative|$($_.Length)|$($_.LastWriteTimeUtc.Ticks)"
                    }
                }
        }
    }
    return (($entries | Sort-Object) -join "`n")
}

Set-Location $Repo
if (-not (Test-Path (Join-Path $Repo ".git"))) { throw "Not a Git repository: $Repo" }
$branch = (Invoke-Git @("branch", "--show-current") | Select-Object -First 1).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Current branch is '$branch', expected '$ExpectedBranch'. Switch to $ExpectedBranch first. No changes were made."
}
$remoteUrl = (Invoke-Git @("remote", "get-url", $Remote) | Select-Object -First 1).Trim()
if ($remoteUrl -notmatch 'NIPUN0BUGS/sinhala-english-llm-from-scratch(\.git)?$') {
    throw "Remote '$Remote' does not match the expected LLM repository: $remoteUrl. No changes were made."
}

# Refuse to start if eligible source/test changes already exist, to avoid committing old work accidentally.
$existing = @(Get-EligibleGitPaths)
if ($existing.Count -gt 0) {
    throw "Eligible source/test changes already exist. Commit or review them manually first. Files:`n$($existing -join "`n")"
}
$staged = @(Invoke-Git @("diff", "--cached", "--name-only") | Where-Object { "$_".Trim() })
if ($staged.Count -gt 0) {
    throw "There are already staged changes. Review/commit them manually before starting. No changes were made."
}

Write-Host "Auto Git push is ACTIVE." -ForegroundColor Green
Write-Host "Repository: $Repo"
Write-Host "Branch: $ExpectedBranch"
Write-Host "Watching only eligible source/test files under src/ and tests/."
Write-Host "Debounce: $DebounceSeconds seconds. Press Ctrl+C to stop."
Write-Host "Data, checkpoints, virtual environments, and secret-named files are excluded."
Write-Host "Safety note: each eligible change will be committed and pushed automatically."

$lastSnapshot = Get-Snapshot
$pendingSince = $null

while ($true) {