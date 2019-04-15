import cv2
import dlib
import numpy as np
shape_predict = dlib.shape_predictor('model/dlib/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib/dlib_face_recognition_resnet_model_v1.dat')

def cal_embed(file_path):
    img = cv2.imread(file_path)
    rec = dlib.rectangle(0, 0, img.shape[1], img.shape[0])
    print(rec)
    shape = shape_predict(img, rec)
    print(shape)
    print(facerec)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    return np.array([elem for elem in face_descriptor])

if __name__ == "__main__":
    img_path = './datasets/19/img_1373.jpg'
    cal_embed(img_path)