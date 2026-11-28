<#
Uruchamia cały projekt w Dockerze: serwery llama.cpp (LLM i embeddingi) oraz aplikację Gradio.

  start.bat              start (pierwszy raz: pobranie modeli i budowa obrazu), potem otwiera przeglądarkę;
                         bez karty NVIDIA modele działają na CPU (wolniej)
  start.bat cpu          start na CPU także wtedy, gdy jest karta NVIDIA
  start.bat stop         zatrzymanie i usunięcie kontenerów (modele, indeks i wyniki zostają)
  start.bat status       stan kontenerów
  start.bat logs         logi na żywo (Ctrl+C kończy podgląd)
  start.bat train        fine-tuning HerBERT w kontenerze (gdy brak models/herbert-faq), potem restart aplikacji
  start.bat test         testy jednostkowe (pytest) w kontenerze

Wymagania: Docker Desktop z backendem WSL2; karta NVIDIA z aktualnym sterownikiem albo około 12 GB RAM dla Dockera.
#>
param(
    [ValidateSet("start", "cpu", "stop", "status", "logs", "train", "test")]
    [string]$Command = "start"
)

Set-Location -Path $PSScriptRoot
$LlamaImage = "ghcr.io/ggml-org/llama.cpp:server-cuda-b12808"
$GgufFiles = @(
    @{ Path = "lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf"; Size = 6975879008 },
    @{ Path = "lmstudio-community/gemma-4-12B-it-QAT-GGUF/mmproj-gemma-4-12B-it-QAT-BF16.gguf"; Size = 175115328 },
    @{ Path = "gpustack/bge-m3-GGUF/bge-m3-Q8_0.gguf"; Size = 634553760 }
)

function Say([string]$Text) { Write-Host "[projekt] $Text" }
function Fail([string]$Text) { Write-Host "[projekt] BŁĄD: $Text" -ForegroundColor Red; exit 1 }

function Read-DotEnv {
    $vars = @{}
    if (Test-Path ".env") {
        foreach ($line in Get-Content ".env" -Encoding UTF8) {
            if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { $vars[$Matches[1]] = $Matches[2] }
        }
    }
    return $vars
}

function Test-GgufDir([string]$Dir) {
    foreach ($f in $GgufFiles) {
        $p = Join-Path $Dir $f.Path
        if (-not (Test-Path $p) -or (Get-Item $p).Length -ne $f.Size) { return $false }
    }
    return $true
}

function Test-Engine { docker info *> $null; return ($LASTEXITCODE -eq 0) }

function Ensure-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Fail "Nie znaleziono Dockera. Zainstaluj Docker Desktop: https://docs.docker.com/desktop/setup/install/windows-install/"
    }
    if (Test-Engine) { return }
    Say "Docker Desktop nie działa, uruchamiam..."
    docker desktop start *> $null
    if ($LASTEXITCODE -ne 0) {
        $exe = @("$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
                 "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($exe) { Start-Process -FilePath $exe }
    }
    for ($i = 0; $i -lt 90; $i++) {
        if (Test-Engine) { return }
        Start-Sleep -Seconds 2
    }
    Fail "Silnik Dockera nie wystartował. Uruchom Docker Desktop ręcznie i spróbuj ponownie."
}

function Test-Gpu {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Say "Nie znaleziono karty NVIDIA (brak nvidia-smi)."
        return $false
    }
    Say "Sprawdzam dostęp do GPU w kontenerach..."
    docker image inspect $LlamaImage *> $null
    if ($LASTEXITCODE -ne 0) {
        Say "Pobieram obraz serwera llama.cpp (CUDA, około 2,6 GB, jednorazowo)..."
        docker pull $LlamaImage | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail "Nie udało się pobrać obrazu $LlamaImage." }
    }
    $gpu = docker run --rm --gpus all --entrypoint nvidia-smi $LlamaImage -L 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($gpu | Out-String)
        Say ("Kontenery nie widzą karty NVIDIA. Do pracy na GPU potrzebne są: Docker Desktop z backendem WSL2 " +
             "(Settings > General > Use the WSL 2 based engine) i aktualny sterownik NVIDIA.")
        return $false
    }
    Say "GPU: $(($gpu | Select-Object -First 1))"
    return $true
}

function Use-Cpu {
    $env:COMPOSE_FILE = "docker-compose.yml;docker-compose.cpu.yml"
    Say "Tryb CPU: modele działają w pamięci RAM. Odpowiedź trwa około 30 s, pytanie z obrazem około 1,5 min."
    $memGb = [math]::Round([double](docker info --format "{{.MemTotal}}") / 1GB, 1)
    if ($memGb -lt 11.5) {
        Write-Host ("[projekt] UWAGA: Docker ma do dyspozycji $memGb GB RAM, a tryb CPU potrzebuje około 12 GB. " +
                    "Zwiększ pamięć w pliku $env:USERPROFILE\.wslconfig (sekcja [wsl2], wiersz memory=12GB), " +
                    "potem wykonaj wsl --shutdown i uruchom projekt ponownie.") -ForegroundColor Yellow
    }
}

function Select-Mode([bool]$ForceCpu) {
    $env:COMPOSE_FILE = "docker-compose.yml"
    if ($ForceCpu) { Use-Cpu; return }
    if (-not (Test-Gpu)) { Use-Cpu }
}

function Ensure-Models {
    $envVars = Read-DotEnv
    if ($env:GGUF_DIR) { $dir = $env:GGUF_DIR }
    elseif ($envVars["GGUF_DIR"]) { $dir = $envVars["GGUF_DIR"] }
    else {
        $dir = "./models/gguf"
        $lmStudio = Join-Path $env:USERPROFILE ".lmstudio\models"
        if (-not (Test-GgufDir $dir) -and (Test-GgufDir $lmStudio)) {
            $dir = $lmStudio -replace '\\', '/'
            Add-Content -Path ".env" -Value "`nGGUF_DIR=$dir" -Encoding UTF8
            Say "Używam modeli z LM Studio: $dir (zapisane w .env jako GGUF_DIR, bez ponownego pobierania)"
        }
    }
    $env:GGUF_DIR = $dir
    if (Test-GgufDir $dir) { Say "Modele GGUF: $dir"; return }
    Say "Brak modeli GGUF w $dir. Pobieram z Hugging Face (około 7,8 GB, jednorazowo)..."
    docker compose --profile setup run --rm --build models
    if ($LASTEXITCODE -ne 0 -or -not (Test-GgufDir $dir)) { Fail "Pobieranie modeli nie powiodło się." }
}

function Get-Health([string]$Service) {
    $id = docker compose ps -q $Service 2>$null
    if (-not $id) { return "brak" }
    $state = docker inspect -f "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}" $id 2>$null
    return "$state".Trim()
}

function Wait-Ready {
    Say "Czekam, aż modele się załadują i aplikacja wystartuje (pierwszy start: 1-3 min)..."
    $deadline = (Get-Date).AddMinutes(15)
    $last = ""
    while ((Get-Date) -lt $deadline) {
        $states = "llm: $(Get-Health 'llm') | embeddings: $(Get-Health 'embeddings') | app: $(Get-Health 'app')"
        if ($states -ne $last) { Say $states; $last = $states }
        if ($states -match "exited|dead") {
            docker compose logs --tail 40
            Fail "Któryś kontener się zatrzymał, logi powyżej."
        }
        if ((Get-Health "app") -eq "running healthy") { return }
        Start-Sleep -Seconds 3
    }
    docker compose logs --tail 40
    Fail "Aplikacja nie wystartowała w ciągu 15 minut, logi powyżej."
}

function Start-Project([bool]$ForceCpu = $false) {
    Ensure-Docker
    if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Say "Utworzono .env z .env.example" }
    Select-Mode $ForceCpu
    Ensure-Models
    if (-not (Test-Path "models/herbert-faq/config.json")) {
        Say "Brak wytrenowanego HerBERT (models/herbert-faq): aplikacja użyje klasyfikatora TF-IDF."
        Say "Fine-tuning HerBERT w kontenerze: start.bat train"
    }
    Say "Buduję obraz aplikacji i uruchamiam kontenery..."
    docker compose up -d --build
    if ($LASTEXITCODE -ne 0) { Fail "docker compose up nie powiodło się." }
    $envVars = Read-DotEnv
    $port = if ($envVars["APP_PORT"]) { $envVars["APP_PORT"] } else { "7860" }
    $llmPort = if ($envVars["LLM_PORT"]) { $envVars["LLM_PORT"] } else { "8080" }
    $embPort = if ($envVars["EMBEDDING_PORT"]) { $envVars["EMBEDDING_PORT"] } else { "8081" }
    $url = "http://localhost:$port"
    Wait-Ready
    Say "Gotowe: $url"
    Say "API modeli: LLM http://localhost:$llmPort/v1, embeddingi http://localhost:$embPort/v1"
    Say "Zatrzymanie: start.bat stop"
    Start-Process $url
}

switch ($Command) {
    "start"  { Start-Project }
    "cpu"    { Start-Project $true }
    "stop"   { Ensure-Docker; docker compose down; Say "Zatrzymano. Modele, indeks i wyniki zostają na dysku." }
    "status" { Ensure-Docker; docker compose ps }
    "logs"   { Ensure-Docker; docker compose logs -f --tail 50 }
    "train"  {
        Ensure-Docker
        Say "Fine-tuning HerBERT (8 epok) w kontenerze aplikacji, na CPU trwa to około 5 minut..."
        docker compose run --rm --build --no-deps app python scripts/train_classifier.py train
        if ($LASTEXITCODE -ne 0) { Fail "Trening nie powiódł się." }
        if ((Get-Health "app") -like "running*") { docker compose restart app; Say "Aplikacja uruchomiona ponownie z HerBERT." }
    }
    "test"   { Ensure-Docker; docker compose run --rm --build --no-deps app python -m pytest -q }
}
