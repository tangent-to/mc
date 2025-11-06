# npm Publishing Setup Guide

Quick guide to set up automated npm publishing for `@tangent.to/mc`.

## Step 1: Create npm Token

1. Log in to npm: https://www.npmjs.com/login
2. Go to Access Tokens: https://www.npmjs.com/settings/YOUR_USERNAME/tokens
3. Click "Generate New Token" > "Automation"
4. Copy the token (starts with `npm_`)

## Step 2: Add Token to GitHub

1. Go to repository settings: https://github.com/tangent-to/mc/settings/secrets/actions
2. Click "New repository secret"
3. Name: `NPM_TOKEN`
4. Value: Paste the npm token from Step 1
5. Click "Add secret"

## Step 3: Verify Setup

1. Go to Actions: https://github.com/tangent-to/mc/actions
2. You should see two workflows:
   - "Publish to npm" (runs on releases)
   - "Test" (runs on push/PR)

## Step 4: First Release

```bash
# On your local machine, main branch
git checkout main
git pull

# Update version
npm version 0.2.0

# Push with tags
git push origin main --tags
```

Then create a GitHub release:
- Go to: https://github.com/tangent-to/mc/releases/new
- Select tag: v0.2.0
- Title: v0.2.0
- Add release notes
- Click "Publish release"

The package will automatically publish to:
https://www.npmjs.com/package/@tangent.to/mc

## Verify Installation

After publishing, test installation:

```bash
# Node.js
npm install @tangent.to/mc

# Deno
import { Model } from "npm:@tangent.to/mc";

# Observable
import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/src/browser.js")
```

## Troubleshooting

**Error: "You do not have permission to publish"**
- Ensure you're a member of the `tangent.to` npm organization
- Check the token has publish permissions
- Verify `publishConfig.access` is set to "public" in package.json

**Error: "Package name too similar"**
- Shouldn't happen with scoped packages
- If it does, contact npm support

**Workflow not running**
- Check that NPM_TOKEN secret is set
- Verify workflows are in `.github/workflows/`
- Check Actions tab for errors

## Security Notes

- Keep npm token secret
- Use "Automation" token type (not personal)
- Rotate tokens every 90 days
- Enable 2FA on npm account

## Support

Questions? Check:
- Full guide: `.github/RELEASE.md`
- GitHub Actions logs: https://github.com/tangent-to/mc/actions
- Open issue: https://github.com/tangent-to/mc/issues
