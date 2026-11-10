#!/bin/sh
# Downloads the uv binaries that the installers ship. The app uses uv to install Python and PyTorch on first launch.
set -e
cd "$(dirname "$0")"
V=$(curl -sI https://github.com/astral-sh/uv/releases/latest | grep -i '^location' | sed 's#.*/tag/##' | tr -d '\r')
T=$(mktemp -d); mkdir -p bin/mac-arm64 bin/mac-x64 bin/win-x64
for pair in aarch64-apple-darwin:mac-arm64 x86_64-apple-darwin:mac-x64; do
  t=${pair%%:*}; d=${pair##*:}
  curl -sL "https://github.com/astral-sh/uv/releases/download/$V/uv-$t.tar.gz" | tar -xz -C "$T"; cp "$T/uv-$t/uv" "bin/$d/"
done
curl -sL -o "$T/w.zip" "https://github.com/astral-sh/uv/releases/download/$V/uv-x86_64-pc-windows-msvc.zip"; unzip -oq "$T/w.zip" -d "$T/w"; cp "$T/w/uv.exe" bin/win-x64/
echo "uv $V"
