from ultralytics import YOLO
import cv2

# YOLO 모델 불러오기
# model = YOLO("/home/user/Desktop/mobis/yolov8/runs/detect/1119/[1119_Icon_dataset]1280_16_11s_1119_Icon/weights/best.pt")  # 사용할 YOLO 모델 크기 선택(n ~ x)
# model = YOLO("/home/user/Desktop/mobis/yolov8/runs/detect/1119/[1119_Icon_dataset+AndStudio]1280_16_11s_1119_Icon/weights/best.pt")  # 사용할 YOLO 모델 크기 선택(n ~ x)

# # 예측 및 결과 저장
# result = model.predict("./test_data/test.png", save=True, conf=0.1)

# # 첫 번째 결과 플롯 저장
# plots = result[0].plot()  # 결과 플롯 얻기
# output_path = "output.png"  # 저장할 경로 설정
# cv2.imwrite(output_path, plots)  # 결과 이미지를 파일로 저장


### ============================== ###
# model_="yolo11s.pt"
# # 320, 128 / 640, 128
# imgsz=640
# batch=128
# name=f"1119/[AndStudio]yolo11s.pt_size320_batch128"

# model = YOLO(f"/home/user/Desktop/mobis/yolov8/runs/detect/{name}/weights/best.pt")  # 사용할 YOLO 모델 크기 선택(n ~ x)
# result = model.predict("./test_data/api33.png", save=True, conf=0.1)
# plots = result[0].plot()  # 결과 플롯 얻기
# output_path = f"./1119_detect_result/{name}.png"  # 저장할 경로 설정
# cv2.imwrite(output_path, plots)  # 결과 이미지를 파일로 저장


model_="yolo11m.pt"
# 320, 128 / 640, 128
imgsz=320
batch=128
name=f"1120/[Only_AndStudio]yolo11s.pt_size320_batch128"

model = YOLO(f"/home/user/Desktop/mobis/yolov8/runs/detect/{name}/weights/last.pt")  # 사용할 YOLO 모델 크기 선택(n ~ x)
result = model.predict("./test_data/api33.png", save=True, conf=0.6, iou=0.1)
plots = result[0].plot()  # 결과 플롯 얻기
output_path = f"./1119_detect_result/{name}.png"  # 저장할 경로 설정
cv2.imwrite(output_path, plots)  # 결과 이미지를 파일로 저장