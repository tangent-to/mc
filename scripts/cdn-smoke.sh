#!/bin/bash
# Check a PUBLISHED release the way a browser notebook loads it: from jsdelivr's +esm bundle, with
# whatever dependency versions jsdelivr pinned into it. Fits a regression with a known answer
# (a = 5, b = 2) in headless Chrome/Chromium, with one chain and with two, and fails otherwise.
# Node and vitest cannot catch a CDN bundling two copies of @tangent.to/grad; this can.
#
#   scripts/cdn-smoke.sh 0.12.1
set -euo pipefail
VERSION=${1:?usage: scripts/cdn-smoke.sh <published version>}
BROWSER=$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)
[ -n "$BROWSER" ] || { echo "no Chrome/Chromium found" >&2; exit 2; }
# A snap-packaged Chromium can only read under ~/snap; use a directory it can reach.
DIR=${SMOKE_DIR:-$([ -d "$HOME/snap/chromium" ] && echo "$HOME/snap/chromium/common/mc-smoke" || mktemp -d)}
mkdir -p "$DIR"
cat > "$DIR/smoke.html" <<EOF
<!DOCTYPE html><html><body><pre id="out">running</pre><script type="module">
const out = document.getElementById("out"), log = (s) => { out.textContent += "\n" + s; };
try {
  const m = await import("https://cdn.jsdelivr.net/npm/@tangent.to/mc@${VERSION}/+esm");
  const mc = m.default ?? m;
  const { add, mul, matmul } = mc.ops;
  const xs = Array.from({ length: 60 }, (_, i) => i / 10), site = xs.map((_, i) => [i % 2, 1 - (i % 2)]);
  const ys = xs.map((x, i) => 5 + 2 * x + (i % 2 ? 0.5 : -0.5) + 0.3 * Math.sin(i));
  const model = new mc.Model("smoke");
  model.addVariable("a", new mc.distributions.Normal(0, 30));
  model.addVariable("b", new mc.distributions.Normal(0, 30));
  model.addVariable("z", new mc.distributions.Normal([0, 0], 1));
  model.addVariable("sigma", new mc.distributions.Lognormal(0, 1));
  model.observe("y", (v) => new mc.distributions.Normal(add(v.a, mul(v.b, xs), matmul(site, v.z)), v.sigma), ys);
  for (const chains of [1, 2]) {
    const fit = await new mc.samplers.NUTS({ stepSize: 0.02 })
      .sample(model, { a: 0, b: 0, z: [0, 0], sigma: 1 }, { chains, nSamples: 300, nWarmup: 300, seed: 1 });
    const mean = (k) => { const x = fit.byChain ? fit.byChain[k].flat() : fit.trace[k]; return x.reduce((s, v) => s + v, 0) / x.length; };
    log(\`chains=\${chains} a=\${mean("a").toFixed(2)} b=\${mean("b").toFixed(3)}\`);
  }
  log("DONE");
} catch (e) { log("ERROR " + e.message); }
</script></body></html>
EOF
RESULT=$("$BROWSER" --headless --disable-gpu --no-sandbox --virtual-time-budget=180000 --dump-dom "file://$DIR/smoke.html" 2>/dev/null \
  | sed -n '/<pre/,/<\/pre>/p' | sed 's/<[^>]*>//g')
echo "$RESULT"
# b is identified to about ±0.02 by these data; a sampler that lost the likelihood lands far off.
echo "$RESULT" | grep -q "^DONE" || { echo "FAIL: did not finish" >&2; exit 1; }
GOOD=$(echo "$RESULT" | LC_ALL=C awk -F'b=' '/^chains=/ { b = $2 + 0; if (b >= 1.9 && b <= 2.1) n++ } END { print n + 0 }')
[ "$GOOD" = 2 ] || { echo "FAIL: mc@${VERSION} does not recover b = 2 from jsdelivr" >&2; exit 1; }
echo "OK: mc@${VERSION} from jsdelivr recovers the known answer"
