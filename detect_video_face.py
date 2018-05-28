from hof.face_detectors import RfcnResnet101FaceDetector
import cv2
import sys


def detect_face(detector, img):
    return detector.detect(img, color=(255, 0, 0), draw_faces=False, min_confidence=0.8)


if len(sys.argv) != 3:
    print('usage: python detect_video_face.py [video_name] [show_image]')
    sys.exit()

video_name = sys.argv[1]
show_image = bool(int(sys.argv[2]))
print('video_name: '+video_name)
print('show_image: '+str(show_image))

cap = cv2.VideoCapture(video_name)
# frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
print(' frame_num: '+str(frame_num))

rfcn_face_detector = RfcnResnet101FaceDetector(min_confidence=0.5)

for i in range(frame_num):
    success, frame = cap.read()
    if not success:
        continue
    faces = detect_face(rfcn_face_detector, frame)
    for k in range(len(faces)):
        f = faces[k]
        face_path = '%s_%05d_%03d_original.jpg' % (video_name, i, k)
        face_roi = frame[f[1]:f[1]+f[3], f[0]:f[0]+f[2]]
        cv2.imwrite(face_path, face_roi)
        print(face_path)
        if show_image:
            cv2.imshow('face_'+str(k), face_roi)
    if show_image:
        cv2.imshow('video', frame)
        if cv2.waitKey(20) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()










