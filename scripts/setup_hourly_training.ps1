# Run once from PowerShell to register the hourly runner at Windows logon.
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
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Run Sinhala-English LLM unit tests and one training epoch every hour." -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Registered and started $TaskName."
Write-Host "The runner logs to $Repo\training-logs and loops every hour while this PC is on."
Write-Host "To stop: Stop-ScheduledTask -TaskName '$TaskName'"
Write-Host "To remove: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:\$false"
