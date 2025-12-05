# Comento_CV

## 현직 AI 엔지니어와 실전 Computer Vision: Git부터 AI 모델링까지

**진행 기간**: 2025.06.30 ~ 2025.07.28  

---

## 프로젝트 개요

4주간 Git 활용부터 이미지 처리, AI 모델 학습, 자율주행 프로토타입 개발까지 **Computer Vision 실무 역량 향상**을 목표로 다양한 과제를 수행하였습니다.

---

## 주차별 업무 요약

### 1주차: Git 및 픽셀 단위 이미지 처리 실습
- GitHub 원격 저장소 생성 및 커밋/브랜치/PR 실습
- OpenCV를 활용한 특정 색상(빨강/파랑/초록) 검출
- 이미지 전처리: 크기 조정, 그레이스케일, 노이즈 제거
- 데이터 증강 및 이상치 필터링 (밝기, 객체 크기 기준)

### 2주차: Unit Test 및 2D → 3D 변환
- `pytest` 기반 유닛 테스트 구성
- 이미지 밝기를 기반으로 Depth Map 생성
- 픽셀 (x, y, 밝기) → 3D 포인트 클라우드 변환
- 시각화를 통한 결과 분석

### 3주차: YOLOv8 기반 객체 탐지 모델 학습
- Roboflow에서 커스텀 데이터셋 활용
- YOLOv8 모델 학습 및 Precision/Recall 추적
- 오분류 케이스 분석 및 Confusion Matrix 시각화
- 학습되지 않은 클래스 이미지 테스트

### 4주차: 차선 인식 자율주행 RC카 개발
- Raspberry Pi + OpenCV 기반 실시간 차선 추출
- ROI 설정, Canny 엣지 검출, Hough Line 검출
- HSV 마스크로 빨간 선 인식 및 중심점 계산
- L9110 모터를 이용한 좌/우회전 RC카 제어

---

## 사용 기술 및 도구

- Python, OpenCV, NumPy, Matplotlib
- Git, GitHub, pytest
- YOLOv8 (Ultralytics)
- Raspberry Pi, L9110 모터 드라이버
- Roboflow, VSCode

---

## 결과 및 회고

이번 프로젝트를 통해 Computer Vision 전반에 걸친 **실무 개발 프로세스 (기획 → 개발 → 테스트 → 시각화 → 하드웨어 적용)** 를 경험할 수 있었습니다.  
특히, 이미지 전처리 및 객체 탐지 과정에서의 문제 해결 경험은 실전 역량 강화에 큰 도움이 되었습니다.

---

## 참고

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Git 공식 문서](https://git-scm.com/book/en/v2)

---

## 벨로그
- 1주차 : https://velog.io/@mdo0421/1%EC%A3%BC%EC%B0%A8ComentoCV
- 1주차 피드백 : https://velog.io/@mdo0421/1%EC%A3%BC%EC%B0%A8ComentoCV%EB%B8%8C%EB%9E%9C%EC%B9%98-%EB%B0%8F-PRPull-Request
- 2주차 : https://velog.io/@mdo0421/2%EC%A3%BC%EC%B0%A8ComentoCVUnit-Test-2D-3D-%EB%B3%80%ED%99%98
- 3주차 : https://velog.io/@mdo0421/3%EC%A3%BC%EC%B0%A8ComentoCVAI-%EA%B8%B0%EB%B0%98-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%AA%A8%EB%8D%B8%EB%A7%81%EB%B0%8FOpenCV%EB%A5%BC%ED%99%9C%EC%9A%A9%ED%95%9C%EA%B2%B0%EA%B3%BC%EC%8B%9C%EA%B0%81%ED%99%94
- 4주차 : https://velog.io/@mdo0421/3%EC%A3%BC%EC%B0%A8ComentoCV%ED%9D%AC%EB%A7%9D%ED%95%98%EB%8A%94-%EC%A0%9C%ED%92%88-%EB%98%90%EB%8A%94-SW-%EC%84%A0%EC%A0%95-%ED%9B%84-%EA%B0%9C%EB%B0%9C-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B8%B0%ED%9A%8D-%EB%B0%8F-%EC%8B%A4%ED%96%89
  
---
