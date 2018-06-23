# -*- coding: utf-8 -*-
import cv2
import dlib


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')


def predict_from_rect(image, l, t, r, b):
    rect = dlib.rectangle(l, t, r, b)
    shp = predictor(image, rect)
    return shp


def detect_and_predict(image):
    if image is None:
        return None
    rect_list = detector(image, 1)
    if rect_list is None:
        return None
    shape_list = []
    for rect in rect_list:
        # l = rect.left()
        # r = rect.right()
        # t = rect.top()
        # b = rect.bottom()
        # print l, t, r, b
        shp = predictor(image, rect)
        shape_list.append(shp)
    return shape_list


img = cv2.imread('test1.jpg')

# shapes = detect_and_predict(img)
# for shape in shapes:
#     for i in range(68):
#         xx = shape.part(i).x
#         yy = shape.part(i).y
#         cv2.circle(img, (xx, yy), 2, (255, 0, 0), 2)
#         cv2.imshow('result', img)

shape = predict_from_rect(img, 170, 313, 491, 634)
for i in range(68):
    xx = shape.part(i).x
    yy = shape.part(i).y
    cv2.circle(img, (xx, yy), 2, (255, 0, 0), 2)
    cv2.imshow('result', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
















