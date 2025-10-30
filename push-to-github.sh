#!/bin/bash

# Easy GitHub Push Script
echo "🚀 GitHub Push Helper"
echo "===================="
echo ""

# Check if remote exists
if git remote get-url origin &> /dev/null; then
    echo "✅ Remote already configured"
    REMOTE_URL=$(git remote get-url origin)
    echo "   Remote: $REMOTE_URL"
    echo ""
    echo "Ready to push? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        git push -u origin main
        echo ""
        echo "✅ Pushed to GitHub!"
        echo "   View at: $REMOTE_URL"
    fi
else
    echo "No remote configured yet."
    echo ""
    echo "Please enter your GitHub username:"
    read -r username
    echo ""
    echo "Repository name (default: flight-metrics-app):"
    read -r repo_name

    if [ -z "$repo_name" ]; then
        repo_name="flight-metrics-app"
    fi

    echo ""
    echo "Creating remote URL: https://github.com/$username/$repo_name.git"
    echo ""

    # Add remote
    git remote add origin "https://github.com/$username/$repo_name.git"

    echo "✅ Remote added!"
    echo ""
    echo "⚠️  IMPORTANT: Before pushing, create the repository on GitHub:"
    echo "   1. Go to: https://github.com/new"
    echo "   2. Repository name: $repo_name"
    echo "   3. DO NOT initialize with README"
    echo "   4. Click 'Create repository'"
    echo ""
    echo "Press ENTER when you've created the repo on GitHub..."
    read -r

    echo "Pushing to GitHub..."
    git branch -M main
    git push -u origin main

    echo ""
    echo "✅ Done! Your code is on GitHub!"
    echo "   View at: https://github.com/$username/$repo_name"
fi
