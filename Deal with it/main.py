import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
specs_ori = cv2.imread('./dealwithit.png', -1)
cap = cv2.VideoCapture(0)

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
     overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
     h, w, _ = overlay.shape
     rows, cols, _ = src.shape
     y, x = pos[0], pos[1]

     for i in range(h):
          for j in range(w):
               if x + i >= rows or y + j >= cols:
                    continue
               alpha = float(overlay[i][j][3] / 255.0)
               src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
     return src
     
def detector(img, classifier):
     faces = classifier.detectMultiScale(img, 1.2, 5, 0, (300, 300), (350, 350))
     for (x, y, w, h) in faces:
          if h > 0 and w > 0:
               glass_symin = int(y + 1.2 * h / 5)
               glass_symax = int(y + 3.0 * h / 5)
               sh_glass = glass_symax - glass_symin
               face_glass_roi_color = img[glass_symin:glass_symax, x:x+w]
               specs = cv2.resize(specs_ori, (w, sh_glass),interpolation=cv2.INTER_CUBIC)
               transparentOverlay(face_glass_roi_color,specs)
     return faces
while 1:
     ret, image = cap.read()
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     fases = detector(image, face_cascade)
     cv2.imshow('2', image)
     k = cv2.waitKey(30) & 0xff
     if k == 27:
          cv2.imwrite('image.jpg', image)
          break
cap.release()
cv2.destroyAllWindows()