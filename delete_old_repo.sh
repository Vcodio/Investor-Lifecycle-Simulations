#!/bin/bash
# Script to delete the old GitHub repository

set -e

OLD_REPO="Lifecycle-Retirement-Simulation"
USERNAME="Vcodio"

# Get token from environment or first argument
if [ -n "$1" ]; then
    GITHUB_TOKEN="$1"
elif [ -z "$GITHUB_TOKEN" ]; then
    echo "============================================================"
    echo "GitHub Personal Access Token Required"
    echo "============================================================"
    echo ""
    echo "To create a token:"
    echo "1. Go to: https://github.com/settings/tokens"
    echo "2. Click 'Generate new token' -> 'Generate new token (classic)'"
    echo "3. Give it a name (e.g., 'repo-deletion')"
    echo "4. Select scopes: 'repo' (full control of private repositories)"
    echo "5. Click 'Generate token' and copy it"
    echo ""
    echo "Usage:"
    echo "  export GITHUB_TOKEN=your_token_here"
    echo "  ./delete_old_repo.sh"
    echo ""
    echo "OR:"
    echo "  ./delete_old_repo.sh your_token_here"
    exit 1
fi

echo "Deleting repository '$USERNAME/$OLD_REPO'..."
DELETE_RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/repos/$USERNAME/$OLD_REPO)

DELETE_HTTP_CODE=$(echo "$DELETE_RESPONSE" | tail -n1)

if [ "$DELETE_HTTP_CODE" = "204" ]; then
    echo "✅ Successfully deleted repository: $USERNAME/$OLD_REPO"
    exit 0
elif [ "$DELETE_HTTP_CODE" = "404" ]; then
    echo "⚠️  Repository '$USERNAME/$OLD_REPO' not found (may already be deleted)"
    exit 0
else
    echo "❌ Error deleting repository (HTTP $DELETE_HTTP_CODE)"
    echo "Response: $(echo "$DELETE_RESPONSE" | sed '$d')"
    echo ""
    echo "You can delete it manually at:"
    echo "https://github.com/$USERNAME/$OLD_REPO/settings"
    exit 1
fi
