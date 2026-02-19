#!/bin/bash

# 1. Check if an argument was provided
if [ -z "$1" ]; then
  echo "Usage: ./release_to_dev.sh [patch|minor|major]"
  exit 1
fi

BUMP_TYPE=$1

# 2. Ensure we are on dev and synchronized
echo "Switching to dev branch..."
git checkout dev
git pull origin dev

# 3. Bump the version
echo "Bumping version ($BUMP_TYPE)..."
NEW_VERSION=$(uv run hatch version "$BUMP_TYPE")

if [ $? -ne 0 ]; then
  echo "Error: Failed to bump version."
  exit 1
fi

# 4. Commit and Tag
echo "Committing version v$NEW_VERSION..."
git commit -am "release: v$NEW_VERSION"

echo "Creating tag v$NEW_VERSION..."
git tag "v$NEW_VERSION"

# 5. Push to dev
echo "Pushing to origin/dev with tags..."
git push origin dev --tags

echo "-------------------------------------------------------"
echo "Done! Version v$NEW_VERSION is now on dev."
echo "GitHub Actions will trigger if set to watch tags."
echo "Remember to merge dev into main when stable."
echo "-------------------------------------------------------"
