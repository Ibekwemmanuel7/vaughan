# Finish the move of the package into vaughan\ (the new files are already there). Run from C:\Users\taylo\milton_da.
Set-Location C:\Users\taylo\milton_da
# 1. remove the old top-level package files (downloads under data\ are untouched)
Remove-Item -Recurse -Force __init__.py, config.py, assimilation, inference, models, physics, scripts, tests, train, __pycache__ -ErrorAction SilentlyContinue
Remove-Item -Force data\__init__.py, data\best_track.py, data\coregistration.py, data\dataset.py, data\download.py, data\synthetic.py -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force data\__pycache__ -ErrorAction SilentlyContinue
# 2. .gitignore: anchor the downloads rule to the repository root so vaughan\data is never ignored
(Get-Content .gitignore) -replace '^data/\*/$', '/data/*/' | Set-Content .gitignore
# 3. stage everything; git records the moves as renames because the contents match
git add -A
git status --short | Select-Object -First 60
