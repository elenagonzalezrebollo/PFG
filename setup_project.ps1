# Variables
$downloadUrl = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt"
$destinationPath = ".\src\models\yolov8x.pt"

$folderPath = Split-Path -Parent $destinationPath
if (-Not (Test-Path -Path $folderPath)) {
    New-Item -ItemType Directory -Force -Path $folderPath
}

# # Download the file
Invoke-WebRequest -Uri $downloadUrl -OutFile $destinationPath

# Install the requirements
pip install -r requirements.txt

$pipShowKivymd = pip show kivymd
if ($pipShowKivymd -eq $null) {
    Write-Output "kivymd no se instaló correctamente. Intentando instalar kivymd nuevamente..."
    pip install kivymd
}

Write-Output "Script completado exitosamente."

$stop = Read-Host "Presiona Enter para salir."