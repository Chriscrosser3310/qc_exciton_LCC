param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Link,

    [ValidateSet("cursor", "code", "none")]
    [string]$Editor = "cursor"
)

function Normalize-LocalLink {
    param([string]$Raw)

    $s = $Raw.Trim()

    # Support markdown link input: [label](url)
    if ($s -match '\(([^)]+)\)$') {
        $s = $matches[1]
    }

    # Strip query/fragment.
    $s = $s -replace '[?#].*$', ''

    # VS Code/Cursor web-file proxy form:
    # https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/.../file.py
    if ($s -match '^https://file\+\.vscode-resource\.vscode-cdn\.net/') {
        $s = $s -replace '^https://file\+\.vscode-resource\.vscode-cdn\.net/', ''
        $s = [System.Uri]::UnescapeDataString($s)
        $s = $s -replace '/', '\'
    }
    elseif ($s -match '^file:///') {
        $u = [System.Uri]$s
        $s = $u.LocalPath
    }

    # Convert /c:/... to c:\... if needed.
    if ($s -match '^\\([A-Za-z]:\\)') {
        $s = $matches[1] + ($s.Substring(4))
    }

    return $s
}

$path = Normalize-LocalLink -Raw $Link

if ($Editor -eq "none") {
    Write-Output $path
    exit 0
}

if ($Editor -eq "cursor") {
    & cursor --goto $path
    exit $LASTEXITCODE
}

& code --goto $path
exit $LASTEXITCODE

