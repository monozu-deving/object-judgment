# 🧱 Column Detector – YOLOv8 기반 기둥 객체 인식 시스템

**Column Detector**는 YOLOv8 세그멘테이션 모델을 기반으로 웹캠 영상에서 기둥 또는 물체를 탐지하고,  
각 객체의 회전각(angle), 상대 좌표(dx, dy), 중심 기준 위치(왼쪽/오른쪽)를 실시간 분석하여 시각화합니다.  
또한 `s` 키를 누를 경우, 해당 프레임과 객체 정보를 이미지 및 CSV로 저장할 수 있습니다.

---

## 📦 기능 요약

- ✅ YOLOv8-seg 기반 실시간 객체 세그멘테이션
- ✅ 객체 중심 좌표 및 회전 각도 계산
- ✅ 가장 중심과 가까운 객체 기준으로 방향 분류:
  - 초록색: 기준 객체
  - 파란색: 기준 객체보다 왼쪽
  - 빨간색: 기준 객체보다 오른쪽
- ✅ `s` 키로 프레임 저장 + CSV 로깅
- ✅ 실행 시 저장 폴더 자동 초기화

---

## 📂 폴더 구조  ##

column_detector/  
├── main.py  
├── configs/  
│ └── settings.py  
├── detection/  
│ ├── init.py  
│ ├── detector.py  
│ └── utils.py  
├── capture/  
│ ├── init.py  
│ └── saver.py  
├── captures/  
│ ├── (저장된 이미지들)  
│ └── detection_log.csv  
───  
---

## ⚙️ 설치 및 실행 방법

### 1. 환경 설정 (Python 3.8+)
```python -m venv venv  
.\venv\Scripts\activate  
pip install ultralytics opencv-python  
```
  
### 2. 실행
```python main.py```
s 키를 누르면 현재 프레임과 객체 정보가 captures/ 폴더에 저장됩니다.
q 키를 누르면 프로그램이 종료됩니다.

📊 저장 형식
captures/capture_0.jpg: 프레임 이미지

captures/detection_log.csv: 객체 정보 로그

CSV 포맷:
filename,x,y,width,height,angle,dx,dy
capture_0.jpg,312,260,52,170,-24.5,5,10  

🔗 사용 모델
yolov8n-seg.pt (Ultralytics YOLOv8 Segmentation model)

보다 정확한 탐지를 원할 경우 yolov8s-seg.pt, yolov8m-seg.pt 등으로 교체하세요.