param(
  [ValidateSet("help", "bootstrap", "test", "example", "mos2", "cmd")]
  [string]$Task = "help",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = "Stop"

$repoWindows = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($repoWindows -notmatch "^[A-Za-z]:\\") {
  throw "Unexpected Windows path format: $repoWindows"
}
$drive = $repoWindows.Substring(0, 1).ToLowerInvariant()
$tail = $repoWindows.Substring(3).Replace("\", "/")
$repoWsl = "/mnt/$drive/$tail"

if ($Task -eq "cmd") {
  if (-not $Args -or $Args.Count -eq 0) {
    throw "For task 'cmd', provide command args. Example: .\scripts\run_in_wsl.ps1 cmd python -m pytest -q"
  }
  & wsl --cd "$repoWsl" bash ./scripts/wsl/run.sh cmd @Args
  exit $LASTEXITCODE
}

& wsl --cd "$repoWsl" bash ./scripts/wsl/run.sh $Task
exit $LASTEXITCODE
