// This PowerShell script will automate the process of preparing and pushing your site to GitHub Pages.
// Save this as deploy.ps1 and run in PowerShell from your gitio directory.

# Rename test.html to index.html if needed
if (Test-Path "test.html") {
    Rename-Item "test.html" "index.html" -Force
}

# Initialize git if not already a repo
if (-not (Test-Path ".git")) {
    git init
}

git add index.html images README.md .gitignore

git commit -m "Deploy to GitHub Pages"

git branch -M main

# Set your GitHub username here
$githubUser = "YOURUSERNAME"
$repoUrl = "https://github.com/$githubUser/$githubUser.github.io.git"

git remote add origin $repoUrl

git push -u origin main

Write-Host "Deployed! Visit https://$githubUser.github.io/ after a few minutes."
