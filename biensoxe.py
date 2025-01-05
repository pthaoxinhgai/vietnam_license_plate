import os
import imutils
import cv2
import numpy as np
import glob
import joblib  # Sử dụng joblib thay vì pickle

def prepocessing(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_removal = cv2.bilateralFilter(gray_img, 12, 30, 30)
    equal_histogram = cv2.equalizeHist(noise_removal)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphology_img = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel)
    edged_img = cv2.Canny(morphology_img, 30, 200)
    return edged_img

def findContours(edged_img):
    contours, hierarchy = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    listContours = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) == 4:
            listContours.append(approx)
    return listContours

def characterSegment(image, lenCon):
    roi_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, (9, 9), 1)
    ret, thre = cv2.threshold(roi_blur, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thre = cv2.dilate(thre, kernel, iterations=1)
    cont, hier = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:lenCon]
    return cont, thre

def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts

def detectCharacter(cropped, model_svm, lenCon):
    listContour, thre = characterSegment(cropped, lenCon)
    plate_info = ""
    for cnt in sort_contours(listContour):
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = h / w
        if 1.2 <= ratio <= 4:
            cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
            child_char = thre[y:y + h, x:x + w]
            child_char = cv2.resize(child_char, dsize=(30, 60))
            child_char = np.array(child_char, dtype=np.float32)
            child_char = child_char.reshape(-1, 30 * 60)
            result = model_svm.predict(child_char)[1]
            result = int(result[0, 0])
            if result <= 9:
                result = str(result)
            elif result >= 65 and result < 91:
                result = chr(result)
            plate_info += result
    return plate_info

def main():
    img_path = "C:/Users/thaop/Downloads/LicensePlateDetection-main/LicensePlateDetection-main/anh/2.jpg"
    if not os.path.exists(img_path):
        print(f"File ảnh {img_path} không tồn tại.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("Không thể mở ảnh. Kiểm tra đường dẫn và định dạng.")
        return

    listImage = [img]
    for img in listImage:
        img = imutils.resize(img, width=800)
        imgPre = prepocessing(img)
        list = findContours(imgPre)
        x, y, w, h = cv2.boundingRect(list[0])
        cropped = img[y + 4:y + h - 2, x + 3:x + w + 2]
        cropped = imutils.resize(cropped, width=500)

        cv2.drawContours(img, list, 0, (0, 255, 0), 4)
        cv2.imshow("anh bien", cropped)
        cv2.waitKey(0)

        # Load mô hình SVM đã train bằng joblib
        try:
            model_svm = joblib.load('C:/Users/thaop/Downloads/LicensePlateDetection-main/LicensePlateDetection-main/svm.xml')
        except Exception as e:
            print(f"Không thể tải mô hình SVM: {e}")
            return

        plate_info = ""
        ratio = w / h
        if 0.8 <= ratio <= 2.8:
            h, w, _ = cropped.shape
            half = h // 2
            top = cropped[:half, :]
            bot = cropped[half:, :]
            plate_info += detectCharacter(top, model_svm, 4)
            plate_info += detectCharacter(bot, model_svm, 6)
        else:
            plate_info += detectCharacter(cropped, model_svm, 10)

        cv2.imshow("anh bien ve contour", cropped)
        cv2.waitKey(0)

        print("Bien so =", plate_info)

if __name__ == "__main__":
    main()
