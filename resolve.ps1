# 1. Grab current User Path
$UserPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
$Sys32Path = "C:\Windows\System32"

# 2. Add System32 back to the User path if it is missing
if ($UserPath -notlike "*$Sys32Path*") {
    $UpdatedPath = "$UserPath;$Sys32Path"
    [System.Environment]::SetEnvironmentVariable("PATH", $UpdatedPath, "User")
    Write-Host "System32 path restored successfully!" -ForegroundColor Green
    Write-Host "CRITICAL: Please CLOSE and REOPEN your terminal for this to work." -ForegroundColor Yellow
} else {
    Write-Host "System32 is already present in User path." -ForegroundColor Yellow
}
