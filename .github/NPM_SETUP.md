# npm Publishing Setup Guide

How automated npm publishing works for `@tangent.to/mc`.

The `Publish to npm` workflow (`.github/workflows/publish.yml`) publishes via
**OIDC trusted publishing** — no long-lived `NPM_TOKEN` secret is required.
On each GitHub release, GitHub mints a short-lived identity that npm trusts, so
there is nothing to rotate or leak.

## Step 1: Configure the trusted publisher on npm (one-time)

1. Log in to npm: https://www.npmjs.com/login
2. Go to the package page: https://www.npmjs.com/package/@tangent.to/mc
3. **Settings → Trusted Publisher → GitHub Actions**, and set:
   - Repository: `tangent-to/mc`
   - Workflow filename: `publish.yml`
4. Save.

The workflow already grants `permissions: id-token: write` and upgrades npm to a
version that supports OIDC (`npm >= 11.5.1`), so no other config is needed.

## Step 2: Cut a release

```bash
# On main, locally
git checkout main && git pull

# Bump the version (also commits + tags)
npm version patch   # or minor / major / 0.4.0

git push origin main --follow-tags
```

Then create a GitHub release for that tag:
- https://github.com/tangent-to/mc/releases/new
- Select the new tag (e.g. `v0.4.0`), add notes, **Publish release**.

Publishing the release triggers the workflow, which runs tests, builds, and
publishes to https://www.npmjs.com/package/@tangent.to/mc with signed provenance.
(`workflow_dispatch` is also available to publish manually from the Actions tab.)

## Verify installation

```bash
# Node.js / bundlers
npm install @tangent.to/mc @tensorflow/tfjs
```

```typescript
// Deno
import { Model } from "npm:@tangent.to/mc";
```

```javascript
// Observable (jsDelivr +esm auto-resolves the tfjs peer dependency)
import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm")
```

## Fallback: token-based publishing

If you ever need to publish without OIDC (e.g. from a machine or a non-trusted
workflow), use a token instead:

1. npmjs.com → **Access Tokens → Generate New Token → Granular Access Token**,
   scoped to the `@tangent.to/mc` package with read/write.
2. Store it as the `NPM_TOKEN` repository secret
   (https://github.com/tangent-to/mc/settings/secrets/actions).
3. In the publish step, drop `--provenance` and add
   `env: { NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }} }`.

Granular tokens are managed from your npm **account**, not from GitHub — GitHub
only stores the value. Rotate them periodically and keep 2FA on the account.

## Troubleshooting

**"You do not have permission to publish"**
- You must be able to publish under the `@tangent.to` scope (org membership /
  publish role). This is the most common cause, not the auth method.

**OIDC publish rejected / "provenance" errors**
- Confirm the trusted publisher on npm matches repo `tangent-to/mc` and workflow
  `publish.yml` exactly.
- Confirm the job keeps `permissions: id-token: write` and runs npm >= 11.5.1.

**Workflow didn't run**
- It triggers on a *published* GitHub release (or manual `workflow_dispatch`),
  not on a plain tag push.

## Support

- Full guide: `.github/RELEASE.md`
- Actions logs: https://github.com/tangent-to/mc/actions
- Issues: https://github.com/tangent-to/mc/issues
