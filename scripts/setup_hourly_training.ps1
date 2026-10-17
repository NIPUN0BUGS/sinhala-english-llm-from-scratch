# Register the hourly runner as a per-user task. Admin rights may be required by some Windows policies.
$ErrorActionPreference = "Stop"
$Repo = "D:\llm\my-llm-git"
$Runner = Join-Path $Repo "scripts\hourly_train_and_test.ps1"
$TaskName = "SinhalaEnglishLLM-Hourly-Train-Test"

if (-not (Test-Path $Runner)) {
    throw "Runner not found at $Runner. Pull the latest branch first."
}
$quotedRunner = [char]34 + $Runner + [char]34
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File $quotedRunner" -WorkingDirectory $Repo
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)

try {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Refresh licensed Sinhala-English training data, run tests, and train one epoch hourly." -Force -ErrorAction Stop | Out-Null
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    Write-Host "Verified task registration: $($task.TaskName)" -ForegroundColor Green
    Write-Host "The runner logs to $Repo\training-logs and loops hourly while this PC is on."
    Write-Host "To stop: Stop-ScheduledTask -TaskName '$TaskName'"
    Write-Host "To remove: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:$false"
}
catch {
    Write-Error "Could not register/start the scheduled task: $($_.Exception.Message)"
    Write-Host "Fallback: keep a normal PowerShell window open and run:" -ForegroundColor Yellow
    Write-Host ('powershell -NoProfile -ExecutionPolicy Bypass -File "' + $Runner + '"') -ForegroundColor Yellow
    exit 1
}
