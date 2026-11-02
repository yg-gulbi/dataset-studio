#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
wheels="${1:?사용법: build_deb.sh <wheel 디렉토리>}"
output="${2:-$root/dist}"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/DEBIAN" "$stage/opt/dataset-studio/wheels" \
  "$stage/usr/bin" "$stage/usr/share/applications" "$stage/usr/share/icons/hicolor/256x256/apps" "$output"
cp "$root/server.py" "$root/native_app.py" "$root/benchmark_20.html" "$stage/opt/dataset-studio/"
cp -r "$root/templates" "$root/static" "$stage/opt/dataset-studio/"
cp "$wheels"/*.whl "$stage/opt/dataset-studio/wheels/"
cp "$root/packaging/control" "$root/packaging/postinst" "$root/packaging/postrm" "$stage/DEBIAN/"
cp "$root/packaging/dataset-studio" "$stage/usr/bin/dataset-studio"
cp "$root/packaging/dataset-studio.desktop" "$stage/usr/share/applications/"
cp "$root/static/app_icon.png" "$stage/usr/share/icons/hicolor/256x256/apps/dataset-studio.png"
chmod 755 "$stage/DEBIAN/postinst" "$stage/DEBIAN/postrm" "$stage/usr/bin/dataset-studio"
dpkg-deb -Znone --build --root-owner-group "$stage" "$output/dataset-studio_1.0.0_amd64.deb"
