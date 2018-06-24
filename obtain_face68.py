import cv2
import dlib
import os


root_dir = '/home/brycezou/DATA/emo_dataset'


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')


def predict_from_rect(image, l, t, r, b):
    rect = dlib.rectangle(l, t, r, b)
    shp = predictor(image, rect)
    return shp


def parse_line(input_line):
    parts1 = input_line.split('.mp4')
    parts2 = parts1[1].split('.jpg')
    index = int(parts2[0][1:6])
    rect = [int(k) for k in parts2[1].replace('\n', '').replace(' ', '', 1).split(' ')]
    path = parts1[0] + '.mp4_' + parts2[0][1:6]
    return path, [index]+rect


rect_map = dict()
with open('face_rect', 'r') as fi:
    lines = fi.readlines()
    for line in lines:
        parser = parse_line(line)
        # print parser
        rect_map[parser[0]] = parser[1]
        assert parser[0] == line[0:len(parser[0])]


def obtain_face68_in_video(video_path, show_image=True):
    print('video_name: '+video_path)
    cap = cv2.VideoCapture(video_path)
    # frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print(' frame_num: '+str(frame_num))

    for i in range(frame_num):
        success, frame = cap.read()
        if not success:
            continue
        face_path_prefix = '%s_%05d' % (video_path, i)

        if face_path_prefix in rect_map:
            rect = rect_map[face_path_prefix]
            shape = predict_from_rect(frame, rect[1], rect[2], rect[3], rect[4])
            with open('face_68', 'a+') as fo:
                fo.write('%s ' % face_path_prefix)
                for p in range(68):
                    xx = shape.part(p).x-rect[1]
                    yy = shape.part(p).y-rect[2]
                    wr = 128.0 / (rect[3]-rect[1])
                    hr = 128.0 / (rect[4]-rect[2])
                    nx = int(xx * wr)
                    ny = int(yy * hr)
                    cv2.circle(frame, (nx, ny), 2, (255, 0, 0), 2)
                    fo.write('%d %d ' % (nx, ny))
                fo.write('\n')
        else:
            pass

        if show_image:
            cv2.imshow('video', frame)
            if cv2.waitKey(5) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


for dir_path, dir_names, file_names in os.walk(root_dir):
    for file_name in file_names:
        if len(file_name) < 5:
            continue
        if file_name[-4:] != '.mp4':
            continue
        org_file_name = file_name
        org_file_path = os.path.join(dir_path, org_file_name)
        obtain_face68_in_video(org_file_path)




























