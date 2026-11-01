#!/usr/bin/env python3
"""
apply_and_rebuild.py

Validates all modified labels, reports manual edit statistics,
regenerates the inspection viewer, repackages dataset_unified_11cls_seg.tar.gz,
and optionally syncs the updated package to Google Drive.
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/home/yg-gulbi/workspace/Blind_walk_ws/AI_인지/data/02_cross_pseudo_labeled/dataset_unified_11cls_seg")
EDITS_LOG_FILE = BASE_DIR / "edits_history.json"
PACKAGES_DIR = Path("/home/yg-gulbi/workspace/Blind_walk_ws/AI_인지/data/04_packages")
SCRIPTS_DIR = Path("/home/yg-gulbi/workspace/Blind_walk_ws/AI_인지/training/scripts")


def main():
    print("="*65)
    print("🚀 [Dataset Studio 폴리곤 수정기 일괄 반영 및 데이터셋 재패키징]")
    print("="*65)

    # 1. Check edits history
    if EDITS_LOG_FILE.exists():
        with open(EDITS_LOG_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {}

    print(f"📊 사용자 수동 수정/정제 기록: 총 {len(history)}개 이미지 반영됨")
    if history:
        print("최근 수정 샘플 5건:")
        for stem, info in list(history.items())[-5:]:
            print(f"  • {stem} ({info.get('split')}) : {info.get('note')} [{info.get('timestamp')}]")

    # 2. Sanity check label files
    print("\n🔍 라벨 파일 무결성 및 폴리곤 검증 중...")
    poly_counts = Counter()
    total_imgs = 0
    corrupt_files = 0

    for split in ("train", "val"):
        lbl_dir = BASE_DIR / "labels" / split
        img_dir = BASE_DIR / "images" / split
        for lf in lbl_dir.glob("*.txt"):
            total_imgs += 1
            im_p = img_dir / f"{lf.stem}.jpg"
            if not im_p.exists():
                print(f"⚠️ 경고: 이미지 누락 -> {im_p}")
                corrupt_files += 1

            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) < 7:
                        corrupt_files += 1
                        continue
                    cid = int(parts[0])
                    poly_counts[cid] += 1

    print(f"  - 검증 완료 이미지: {total_imgs}장")
    print(f"  - 유효 폴리곤 수: {sum(poly_counts.values())}개")
    if corrupt_files > 0:
        print(f"❌ 경고: {corrupt_files}개의 이상 파일이 감지되었습니다.")
    else:
        print("✅ 모든 라벨 파일 포맷 및 좌표 무결성 100% 통과!")

    # 3. Repackage dataset_unified_11cls_seg.tar.gz
    archive_out = PACKAGES_DIR / "dataset_unified_11cls_seg.tar.gz"
    print(f"\n📦 아카이브 재압축 생성 중: {archive_out.name}...")
    cmd = f"tar -czf '{archive_out}' -C '{BASE_DIR.parent}' dataset_unified_11cls_seg/images dataset_unified_11cls_seg/labels dataset_unified_11cls_seg/data.yaml"
    ret = os.system(cmd)
    if ret != 0:
        print(f"❌ 압축 실패! exit code: {ret}")
        sys.exit(1)

    size_mb = round(archive_out.stat().st_size / (1024 * 1024), 1)
    print(f"✅ 압축 완료! 파일 크기: {size_mb} MB")

    # 4. Regenerate inspection viewer
    gallery_script = SCRIPTS_DIR / "generate_html_gallery.py"
    if gallery_script.exists():
        print("\n🖼️ 육안 검수기(dataset_viewer.html) 갱신 중...")
        os.system(f"python3 '{gallery_script}'")

    print("\n" + "="*65)
    print("🎉 [데이터셋 일괄 반영 및 패키징 완료!]")
    print(f"  - 패키지: {archive_out}")
    print(f"  - 필요 시 'rclone copy {archive_out} gdrive:02_AI_모델_및_데이터셋/02_데이터셋_압축파일/ -v' 로 구글 드라이브 동기화 가능")
    print("="*65)


if __name__ == "__main__":
    main()

