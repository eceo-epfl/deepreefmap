#!/usr/bin/env bash
# Wrap the PyApp binary in a .app bundle and pack it into a .dmg (macOS host).
#
# A bare Mach-O is unusable for non-technical users: Finder opens it in
# Terminal, and Gatekeeper on macOS 15+ has no right-click bypass. The bundle
# gives Finder/Dock identity; the dmg gives drag-to-Applications install.
# Unsigned for now — first open needs System Settings > Privacy & Security >
# "Open Anyway" once. Signing/notarization will reuse this same structure.
#
# Usage: make_app_bundle.sh <binary-path> <version> <dmg-output-path>
set -euo pipefail

BINARY="$1"
VERSION="$2"
DMG_OUT="$3"

STAGE=$(mktemp -d)
APP="$STAGE/DeepReefMap.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$BINARY" "$APP/Contents/MacOS/deepreefmap"
chmod +x "$APP/Contents/MacOS/deepreefmap"

# Icon: PNG -> iconset -> icns (sips/iconutil ship with macOS).
ICONSET=$(mktemp -d)/icon.iconset
mkdir -p "$ICONSET"
SRC_PNG="deepreefmap/resources/icon.png"
for size in 16 32 64 128 256 512; do
  sips -z "$size" "$size" "$SRC_PNG" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  sips -z "$((size * 2))" "$((size * 2))" "$SRC_PNG" \
    --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/icon.icns"

# CFBundleVersion must be dot-separated integers; strip branch-build suffixes
# like 1.1.0+g1234abc.
BUNDLE_VERSION=${VERSION%%+*}
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>ch.epfl.eceo.deepreefmap</string>
    <key>CFBundleName</key>
    <string>DeepReefMap</string>
    <key>CFBundleDisplayName</key>
    <string>DeepReefMap</string>
    <key>CFBundleExecutable</key>
    <string>deepreefmap</string>
    <key>CFBundleIconFile</key>
    <string>icon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${BUNDLE_VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${BUNDLE_VERSION}</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

ln -s /Applications "$STAGE/Applications"
mkdir -p "$(dirname "$DMG_OUT")"
rm -f "$DMG_OUT"
hdiutil create -volname DeepReefMap -srcfolder "$STAGE" -ov -format UDZO "$DMG_OUT"
echo "Wrote $DMG_OUT"
