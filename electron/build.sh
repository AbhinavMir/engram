#!/bin/sh
# Builds Engram.app from the local Electron runtime. Usage: ./build.sh [output folder]. Run bun install here and in ../ai first.
set -e
cd "$(dirname "$0")"
OUT="${1:-build}"; APP="$OUT/Engram.app"; RES="$APP/Contents/Resources"
mkdir -p "$OUT"; rm -rf "$APP"
cp -R node_modules/electron/dist/Electron.app "$APP"
mkdir -p "$RES/app" "$RES/engram"
cp main.js preload.js package.json "$RES/app/"
rm -f "$RES/default_app.asar"
cp ../server.py ../engine.py ../index.html "$RES/engram/"
mkdir -p "$RES/engram/docs"; cp ../docs/logo.png "$RES/engram/docs/"
cp icon.icns "$RES/electron.icns"
cp -R ../ai "$RES/engram/ai"
/usr/libexec/PlistBuddy -c "Set :CFBundleName Engram" -c "Set :CFBundleDisplayName Engram" -c "Set :CFBundleIdentifier dev.engram.app" "$APP/Contents/Info.plist"
codesign --force --deep --sign - "$APP" >/dev/null 2>&1
echo "$APP"
