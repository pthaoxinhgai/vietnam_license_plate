import os
import joblib
import cv2
import numpy as np
import glob
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Kích thước ảnh kỹ tự
digit_w = 30
digit_h = 60

def get_digit_data(path):
    digit_list = []
    label_list = []

    # Đọc dữ liệu cho các chữ số từ 0 đến 9
    for number in range(10):
        print(f"Loading data for {number} from {path}{number}")
        images_loaded = False  # Biến để kiểm tra nếu có ảnh được tải
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            if img is not None:
                img = np.array(img)
                img = cv2.resize(img, (digit_w, digit_h))  # Resize ảnh để phù hợp với kích thước
                digit_list.append(img)
                label_list.append(int(number))
                images_loaded = True  # Đánh dấu là đã tải ảnh
        if not images_loaded:
            print(f"Warning: No images found for {number} in {path}{number}")

    # Đọc dữ liệu cho các ký tự A-Z
    for number in range(65, 91):
        char = chr(number)
        print(f"Loading data for {char} from {path}{char}")
        images_loaded = False  # Biến để kiểm tra nếu có ảnh được tải
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            if img is not None:
                img = np.array(img)
                img = cv2.resize(img, (digit_w, digit_h))
                digit_list.append(img)
                label_list.append(ord(char))  # Chuyển ký tự thành mã ASCII
                images_loaded = True  # Đánh dấu là đã tải ảnh
        if not images_loaded:
            print(f"Warning: No images found for {char} in {path}{char}")

    return digit_list, label_list

# Lấy dữ liệu huấn luyện và kiểm tra
path_train = "C:/Users/thaop/Downloads/LicensePlateDetection-main/LicensePlateDetection-main/datatrain/0"
path_test = "C:/Users/thaop/Downloads/LicensePlateDetection-main/LicensePlateDetection-main/datatest/0"

digit_train, label_train = get_digit_data(path_train)
digit_test, label_test = get_digit_data(path_test)

# Kiểm tra nếu không có dữ liệu
if len(digit_train) == 0 or len(digit_test) == 0:
    print("Error: No images loaded for training or testing. Please check your image directories.")
    exit()

# Chuyển dữ liệu thành mảng numpy
digit_train = np.array(digit_train, dtype=np.float32)
digit_test = np.array(digit_test, dtype=np.float32)

# Chuyển đổi kích thước của dữ liệu thành 1 chiều
digit_train = digit_train.reshape(-1, digit_h * digit_w)
digit_test = digit_test.reshape(-1, digit_h * digit_w)

label_train = np.array(label_train)
label_test = np.array(label_test)

# Kiểm tra kích thước và kiểu dữ liệu
print(f"Training data shape: {digit_train.shape}")
print(f"Test data shape: {digit_test.shape}")
print(f"Training labels shape: {label_train.shape}")
print(f"Test labels shape: {label_test.shape}")

# Tạo mô hình LinearSVC và huấn luyện
model = LinearSVC(C=10)
model.fit(digit_train, label_train)

# Dự đoán và tính độ chính xác
y_pred = model.predict(digit_test)
print("Accuracy Score:", accuracy_score(label_test, y_pred))

# Lưu mô hình bằng joblib
xml = "C:/Users/thaop/Downloads/LicensePlateDetection-main/LicensePlateDetection-main/svm_model.pkl"
if os.path.exists(xml):
    os.remove(xml)  # Xóa file cũ nếu có

joblib.dump(model, xml)
print(f"Model saved to {xml}")
