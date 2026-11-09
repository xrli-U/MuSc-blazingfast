$ErrorActionPreference = 'Stop'
$directory = Join-Path $PSScriptRoot '../models/backbone/weights'
$name = 'ViT-L-14-336px.pt'
$hash = '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02'
$url = "https://openaipublic.blob.core.windows.net/clip/models/$hash/$name"
$directory = [IO.Directory]::CreateDirectory($directory).FullName
$target = Join-Path $directory $name
if (Test-Path -LiteralPath $target) {
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $hash) {
        throw "Existing checkpoint has an unexpected SHA-256. Move it aside before retrying: $target"
    }
} else {
    $partial = Join-Path $directory ([IO.Path]::GetRandomFileName())
    try {
        & curl.exe --fail --location --retry 2 --output $partial $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed (curl exit $LASTEXITCODE)." }
        if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $hash) {
            throw 'Downloaded checkpoint has an unexpected SHA-256.'
        }
        [IO.File]::Move($partial, $target)
    } finally {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }
    }
}
Write-Output "Verified $target"
