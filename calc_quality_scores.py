import cv2
import numpy as np
import os
from collections import defaultdict
from shutil import copy
import pandas as pd
from scipy.optimize import fsolve
from preprocess.util import get_api_response, parse_response_dict
from tensorflow.keras.models import load_model


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# face image quality assessment (FIQA) criterion considering image sharpness
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

def get_brenner(img):
    '''
    get the sharpness value of an image based on the Brenner Function
    the clearer the image, the larger the return value
    :param img:narray       
    :return: float 
    '''
    shape = np.shape(img)
    
    out = 0
    for y in range(0, shape[1]):
        for x in range(0, shape[0]-2):
            
            out+=(int(img[x+2,y])-int(img[x,y]))**2
            
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# face image quality assessment (FIQA) criterion considering facial symmetry
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

def get_distance(p1, p2):
    # calculate the distance between two points p1,p2
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5


def get_symmetric_p(m1, m2, p1):
    # given a line by two points m1,m2,
    # get the symmetric point p2 from p1
    p2 = [0, 0]
    # if line (m1,m2) is vertical
    if m1[0] == m2[0]:
        p2[0] = 2*m1[0] - p1[0]
        p2[1] = p1[1]
    else:
        def func(x):
            return [(m1[0]-m2[0])*(p1[1]+x[1]-2*m1[1])-(m1[1]-m2[1])*(p1[0]+x[0]-2*m1[0]),
                    (m1[1]-m2[1])*(p1[1]-x[1])-(m1[0]-m2[0])*(x[0]-p1[0])]
        root = fsolve(func, [1, 1])
        p2 = root.tolist()
    return tuple(p2)


def calc_symmetry(landmarks):
    nose_bridge1 = (landmarks['nose_bridge1']['x'], landmarks['nose_bridge1']['y'])
    nose_middle_contour = (landmarks['nose_middle_contour']['x'], landmarks['nose_middle_contour']['y'])
    contour_left = (landmarks['contour_left1']['x'], landmarks['contour_left1']['y'])
    contour_right = (landmarks['contour_right1']['x'], landmarks['contour_right1']['y'])
    lefteye_inner = (landmarks['left_eye_right_corner']['x'], landmarks['left_eye_right_corner']['y'])
    lefteye_outter = (landmarks['left_eye_left_corner']['x'], landmarks['left_eye_left_corner']['y'])
    righteye_inner = (landmarks['right_eye_left_corner']['x'], landmarks['right_eye_left_corner']['y'])
    righteye_outter = (landmarks['right_eye_right_corner']['x'], landmarks['right_eye_right_corner']['y'])
    d1 = get_distance(contour_right, get_symmetric_p(nose_bridge1,nose_middle_contour,contour_left))
    d2 = get_distance(righteye_inner, get_symmetric_p(nose_bridge1,nose_middle_contour,lefteye_inner))
    d3 = get_distance(righteye_outter, get_symmetric_p(nose_bridge1,nose_middle_contour,lefteye_outter))
    return (d1+d2+d3)/3


def calc_symmetry_main(src_path):
    # get the symmetry value of a face image
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    key = "xtLLC50hhaU98vVu4imNc38kLDxhCw05"
    secret = "lutjiHEBUKUwREupNUIURzV_kGLWccpM"
    un_detected_face = []
    try:
        response_dict = get_api_response(http_url, key, secret, src_path)
    except:
        print("Re-access to the face detect API!")
        response_dict = get_api_response(http_url, key, secret, src_path)
    landmarks = parse_response_dict(response_dict)
    if landmarks is None:
        un_detected_face.append(src_path)
    val_sym = calc_symmetry(landmarks)
    if len(un_detected_face) > 0:
        print("undetected faces:", un_detected_face)
    
    return val_sym



# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# face image quality assessment (FIQA) criterion by the FaceQnet model 
# taking as a reference: https://github.com/uam-biometrics/FaceQnet
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

def load_data(src_dir):
    # load the images and the name of images
	X_test = []
	image_names = []
	print('Read test images')
	
	for imagen in [imagen for imagen in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, imagen))]:
		imagenes = os.path.join(src_dir, imagen)
		print(imagenes)
		img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (224, 224))
		X_test.append(img)
		image_names.append(imagenes)
	return X_test, image_names


def normalize_data(src_dir):
    # normalize the image pixel values to floats
    image_data, image_names = load_data(src_dir)
    image_data = np.array(image_data, copy=False, dtype=np.float32)
    return image_data, image_names


def calc_faceqnet_score(src_dir, dst_file, model_path):
    model = load_model(model_path)
    y, image_names = normalize_data(src_dir)
    score = model.predict(y, batch_size=1, verbose=1)
    predictions = score

    dict_score = defaultdict(float)
    cnt = 0
    for score in predictions:
        dict_score[image_names[cnt]] = score[0]
        cnt += 1
    d_order=sorted(dict_score.items(),key=lambda x:x[1],reverse=False)

    with open(dst_file,'w') as f:
        # Save the scores in a .txt file
        f.write("img;score\n")
        for item in d_order:
            f.write("%s" % item[0])
            prediction_score = item[1]
            # Constrain the output scores to the 0-1 range, 
            # 0:worst quality, 1:best quality
            if float(prediction_score)<0:
                prediction_score='0'
            elif float(prediction_score)>1:
                prediction_score='1'
            f.write(";%s\n" % prediction_score)