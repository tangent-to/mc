# Release and Deployment Guide

This document describes how to publish `@tangent.to/mc` to npm.

## Prerequisites

### 1. npm Account Setup

1. Create an npm account at https://www.npmjs.com/signup
2. Join or create the `tangent.to` organization at https://www.npmjs.com/org/tangent.to
3. Ensure you have publish permissions for `@tangent.to` scope

### 2. Generate npm Token

1. Go to https://www.npmjs.com/settings/YOUR_USERNAME/tokens
2. Click "Generate New Token"
3. Select "Automation" token type (for CI/CD)
4. Copy the token (starts with `npm_...`)

### 3. Configure GitHub Secret

1. Go to your repository: https://github.com/tangent-to/mc
2. Navigate to Settings > Secrets and variables > Actions
3. Click "New repository secret"
4. Name: `NPM_TOKEN`
5. Value: Paste your npm token
6. Click "Add secret"

## Publishing Methods

### Method 1: Publish via GitHub Release (Recommended)

This is the recommended method as it creates a git tag and GitHub release.

1. **Update version in package.json**:
   ```bash
   npm version patch  # or minor, or major
   # This creates a git tag like v0.2.1
   ```

2. **Push the tag**:
   ```bash
   git push origin main --tags
   ```

3. **Create GitHub Release**:
   - Go to https://github.com/tangent-to/mc/releases/new
   - Choose the tag you just pushed
   - Title: `v0.2.1` (match the version)
   - Description: Add release notes (see template below)
   - Click "Publish release"

4. **Automatic Publishing**:
   - GitHub Actions will automatically run
   - Tests will be executed
   - Documentation will be generated
   - Package will be published to npm
   - Check the Actions tab for progress

### Method 2: Manual Workflow Dispatch

For testing or urgent releases:

1. Go to Actions tab: https://github.com/tangent-to/mc/actions/workflows/publish.yml
2. Click "Run workflow"
3. Select branch (usually `main`)
4. Optionally specify version (e.g., `0.2.1`)
5. Click "Run workflow"

### Method 3: Local Publishing (Emergency Only)

If GitHub Actions are down:

```bash
# Ensure you're on main branch and up to date
git checkout main
git pull

# Update version
npm version patch

# Run tests
npm run test:pymc

# Generate docs
npm run docs:generate

# Login to npm (interactive)
npm login

# Publish
npm publish --access public

# Push changes
git push origin main --tags
```

## Release Notes Template

When creating a GitHub release, use this template:

```markdown
## What's New in v0.2.1

### Features
- Description of new features
- Another feature

### Improvements
- Performance improvements
- Better error messages

### Bug Fixes
- Fixed issue with GP predictions
- Resolved memory leak in samplers

### Documentation
- Updated Observable guide
- Added new examples

### Breaking Changes
- None

## Installation

```bash
npm install @tangent.to/mc@0.2.1
```

## Usage

```javascript
import { Model, Normal, MetropolisHastings } from '@tangent.to/mc';
// Your code here
```

See [documentation](https://github.com/tangent-to/mc#readme) for more details.
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.2.0): New features, backward compatible
- **Patch** (0.2.1): Bug fixes, backward compatible

Use npm version commands:

```bash
npm version major   # 0.2.0 -> 1.0.0
npm version minor   # 0.2.0 -> 0.3.0
npm version patch   # 0.2.0 -> 0.2.1
```

## Pre-release Versions

For beta or alpha releases:

```bash
npm version prerelease --preid=beta  # 0.2.0 -> 0.2.1-beta.0
npm publish --tag beta
```

Install with:
```bash
npm install @tangent.to/mc@beta
```

## Verification After Publishing

1. **Check npm**: Visit https://www.npmjs.com/package/@tangent.to/mc
2. **Test installation**:
   ```bash
   npm install @tangent.to/mc@latest
   ```
3. **Verify in Observable**:
   ```javascript
   import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/src/browser.js")
   ```
4. **Test in Deno**:
   ```typescript
   import { Model } from "npm:@tangent.to/mc";
   ```

## Troubleshooting

### "You do not have permission to publish"

- Ensure you're a member of the `tangent.to` organization
- Check that `NPM_TOKEN` secret is set correctly
- Verify the token has publish permissions

### "Version already exists"

- You're trying to publish an existing version
- Update version in package.json
- Run `npm version patch` to increment

### "Package name too similar to existing packages"

- This shouldn't happen with scoped packages
- Contact npm support if it does

### Tests failing in CI

- Check the Actions logs for details
- Run tests locally: `npm run test:pymc`
- Ensure all dependencies are compatible

### Documentation not generated

- Check that `scripts/generate-docs.js` exists
- Run locally: `npm run docs:generate`
- Verify JSDoc comments are properly formatted

## Rollback Procedure

If a bad version was published:

1. **Deprecate the version**:
   ```bash
   npm deprecate @tangent.to/mc@0.2.1 "This version has critical bugs"
   ```

2. **Publish a fixed version**:
   ```bash
   npm version patch  # 0.2.1 -> 0.2.2
   # Fix the issues
   npm publish
   ```

3. **Update documentation**: Notify users of the issue

## Security

- **Never commit** npm tokens to the repository
- **Rotate tokens** regularly (every 90 days)
- **Use automation tokens** for CI/CD (not personal tokens)
- **Enable 2FA** on your npm account

## Support

If you encounter issues:

1. Check GitHub Actions logs
2. Review this documentation
3. Check npm publish logs
4. Open an issue: https://github.com/tangent-to/mc/issues

## Checklist Before Release

- [ ] All tests passing locally (`npm run test:pymc`)
- [ ] Documentation generated (`npm run docs:generate`)
- [ ] Examples run successfully
- [ ] Version number updated in package.json
- [ ] CHANGELOG.md updated (if exists)
- [ ] Release notes prepared
- [ ] No uncommitted changes
- [ ] On main branch and up to date

## Automated Checks

The publish workflow automatically:

1. Runs PyMC comparison tests
2. Generates API documentation
3. Verifies package name is `@tangent.to/mc`
4. Publishes with `--access public` flag
5. Creates a deployment summary

## Post-Release Tasks

After a successful release:

1. Announce on relevant channels
2. Update README if needed
3. Close resolved issues
4. Update project board
5. Plan next release

## Contact

For questions about releasing:
- Open an issue: https://github.com/tangent-to/mc/issues
- Check Actions logs: https://github.com/tangent-to/mc/actions
