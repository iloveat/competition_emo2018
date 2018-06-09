from hof.face_detectors import RfcnResnet101FaceDetector
import cv2
import sys


def detect_face(detector, img):
    return detector.detect(img, color=(255, 0, 0), draw_faces=False, min_confidence=0.8)


def find_largest_face(face_list):
    largest_face = []
    max_face_area = -1
    max_face = None
    for kk in range(len(face_list)):
        ff = face_list[kk]
        if ff[2] * ff[3] > max_face_area:
            max_face_area = ff[2] * ff[3]
            max_face = ff
    if max_face is not None:
        largest_face.append(max_face)
    return largest_face


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
    faces = find_largest_face(faces)
    for k in range(len(faces)):
        f = faces[k]
        face_path_org = '%s_%05d_%03d_org.jpg' % (video_name, i, k)
        face_roi_org = frame[f[1]:f[1]+f[3], f[0]:f[0]+f[2]]
        cv2.imwrite(face_path_org, face_roi_org)

        face_path_pad = '%s_%05d_%03d_pad.jpg' % (video_name, i, k)
        pad_t = f[1]-f[3]/10 if f[1]-f[3]/10 > 0 else 0
        pad_b = f[1]+f[3]+f[3]/10 if f[1]+f[3]+f[3]/10 < frame.shape[0] else frame.shape[0]
        pad_l = f[0]-f[2]/10 if f[0]-f[2]/10 > 0 else 0
        pad_r = f[0]+f[2]+f[2]/10 if f[0]+f[2]+f[2]/10 < frame.shape[1] else frame.shape[1]
        face_roi_pad = frame[pad_t:pad_b, pad_l:pad_r]
        cv2.imwrite(face_path_pad, face_roi_pad)

        print(face_path_org)
        if show_image:
            cv2.imshow('face_'+str(k), face_roi_org)
    if show_image:
        cv2.imshow('video', frame)
        if cv2.waitKey(20) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()










