import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model3.h5')

index_to_shape = {0: 'circle', 1: 'square', 2: 'triangle'}

window_name = 'Draw'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 500, 500)

image = np.ones((500, 500), dtype="uint8") * 255

drawing = False
pt1_x, pt1_y = None, None


def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(image, (pt1_x, pt1_y), (x, y), color=(0, 0, 0), thickness=27)
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(image, (pt1_x, pt1_y), (x, y), color=(0, 0, 0), thickness=27)


cv2.setMouseCallback(window_name, line_drawing)

while 1:
    cv2.imshow(window_name, image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        image = np.ones((500, 500), dtype="uint8") * 255
    if key == ord('p'):
        img = cv2.resize(image, (28, 28))  # Измените размер на соответствие входу вашей модели
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)  # Измените форму на соответствие входу вашей модели
        prediction = model.predict(img)
        predicted_index = np.argmax(prediction)
        print('Predicted shape:', index_to_shape[predicted_index])
        print('Confidence:', np.max(prediction))

    elif key == 27:
        break

cv2.destroyAllWindows()

