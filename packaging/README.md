# Ubuntu 패키지

- 대상: Ubuntu 24.04, x86-64
- 설치: `sudo apt install ./dataset-studio_1.0.0_amd64.deb`
- 실행: 앱 목록에서 **데이터셋 구축 스튜디오** 선택
- 제거: `sudo apt remove dataset-studio` (`purge` 시 설치된 Python 환경도 제거)
- 사용자 데이터: `~/.local/share/dataset-studio/` (`XDG_DATA_HOME` 설정 시 해당 경로). 패키지 제거 시 데이터는 보존됩니다.

패키지에 Python AI 라이브러리를 포함하지만 데이터셋과 모델 가중치 `.pt`는 포함하지 않습니다. AI 기능에 사용할 파일은 별도로 준비해야 합니다. NVIDIA GPU를 사용하려면 해당 PyTorch/CUDA 빌드와 호환되는 NVIDIA 드라이버가 필요합니다. 드라이버가 없으면 CPU로 실행합니다.

설치 시험에서 패키지 내부 wheel은 약 3 GiB, 설치된 Python 환경은 약 5.9 GiB를 사용했습니다. 설치 파일 자체와 설치 중 임시 공간도 별도로 필요합니다. Ubuntu 시스템 의존성 설치를 위해 인터넷 연결이 필요할 수 있습니다. Ultralytics 등 포함된 구성 요소에는 각각의 라이선스가 적용됩니다.

CPU 자동 라벨링은 Ubuntu 24.04의 임시 설치 환경에서 시험했습니다. NVIDIA GPU와 다른 PC에서의 실행은 아직 검증하지 않았습니다.

빌드: Ubuntu 24.04 x86-64의 Python 3.12 환경에서 `python3 -m pip download --dest <wheel-dir> 'ultralytics==8.4.166' 'flask==3.1.3' 'PyYAML==6.0.3'`로 의존성을 받은 뒤 `bash packaging/build_deb.sh <wheel-dir>`를 실행합니다.
