import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'images/input/test.jpg'
image = cv2.imread(image_path)

# Kiểm tra xem ảnh có tải thành công không
if image is not None:

    # Chuyển ảnh sang thang độ xám để giảm độ phức tạp
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Làm mượt ảnh bằng bộ lọc Gaussian để giảm nhiễu
    blurred = cv2.GaussianBlur(gray,(5,5), 0)

    # Phát hiện biên bằng thuật toán Canny
    edged = cv2.Canny(blurred, 30, 200)

    # Tìm các đường viền (contours) trong ảnh biên
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    screen_contour = None

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        # Kiểm tra nếu contour có đúng 4 điểm, có thể là hình chữ nhật của biển số
        if len(approx) ==4:
            screnn_contour = approx
            break

    # Tạo một bản sao của ảnh để vẽ lên
    result_image = image.copy()
    if screen_contour is not None:
        # Vẽ đường viền của biển số lên ảnh
        cv2.drawContours(result_image, [screen_contour],-1, (0, 255, 0), 3)

        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [screen_contour], 0, 255, -1)
        plate_image = cv2.bitwise_and(image, image, mask=mask)

        # Tìm tọa độ của vùng biển số trong mặt nạ
        x, y = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped_plate = gray[topx:bottomx+1, topy:bottomy+1]

        plt.figure(figsize=(10,5))

        # Hiển thị ảnh với biển số đã được khoanh vùng
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(result_image,cv2.COLOR_BAYER_BG2BGR))
        plt.title('Detected License Plate')

        # Hiển thị phần biển số đã cắt
        plt.subplot(1, 2, 2)
        plt.imshow(cropped_plate, cmap='gray')
        plt.title('Cropped License Plate')

        plt.show()
    else:
        print('No license plate detected in the image.')  
else:
    print("Image could not be loaded.")          