param(
  [string]$Root = "C:\Users\jonat\OneDrive\Desktop\nobamboozle"
)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "$Root\venv\Scripts\Activate"
$env:NCBI_API_KEY = $env:NCBI_API_KEY  # set in your user env once if you have it
python "$Root\scripts\harvest.py" --terms "$Root\topics.yml" --years 10 --cap 800
