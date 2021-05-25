from mtcnn import MTCNN
import cv2
import os
import numpy as np
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest='mode', type=str, default='train')

args = parser.parse_args()

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def findBestFace(faces):
    if len(faces) == 1:
        return faces[0]['keypoints']
    else:
        ret = faces[0]
        for face in faces:
            if face['confidence'] > ret['confidence']:
                ret = face
        return ret['keypoints']

def genTrain():
    print('\ngenerating training landmark file\n')
    print('will save the file to ./data/train/APDlandmark.txt')

    raw_image_list = np.array(os.listdir('./data/train/A'))
    raw_image_list = np.sort(raw_image_list)
    # get name list and transfer into number label
    name_to_id = {}
    label_list = []
    count = 0
    for path in raw_image_list:
        temp_name = path.split('_')[0]
        if temp_name not in name_to_id:
            name_to_id[temp_name] = count
            count += 1
        label_list.append(name_to_id[temp_name])
    label_list = np.array(label_list)
    print(f'class number: {len(name_to_id)}')

    # pickle name_to_id
    with open('./data/train/name_to_id.pkl', 'wb') as f:
        pickle.dump(name_to_id, f)
    print('finish writing name_to_id.pkl')

    detector = MTCNN()
    face_list = []
    for filename in tqdm(raw_image_list):
        img = cv2.cvtColor(cv2.imread(os.path.join('./data/train/A', filename)), cv2.COLOR_BGR2RGB )
        face_list.append(findBestFace(detector.detect_faces(img)))
        del img

    with open('./data/train/APDlandmark.txt', 'w') as f:
        for i in range(len(label_list)):
            s = f'{raw_image_list[i]}\t{label_list[i]}\t' + \
                f'{face_list[i]["left_eye"][0]}\t{face_list[i]["left_eye"][1]}\t' + \
                f'{face_list[i]["right_eye"][0]}\t{face_list[i]["right_eye"][1]}\t' + \
                f'{face_list[i]["nose"][0]}\t{face_list[i]["nose"][1]}\t' + \
                f'{face_list[i]["mouth_left"][0]}\t{face_list[i]["mouth_left"][1]}\t' + \
                f'{face_list[i]["mouth_right"][0]}\t{face_list[i]["mouth_right"][1]}\n'
            f.write(s)

def genTest(mode):
    print(f'\ngenerating testing landmark file (mode: {mode})\n')
    print(f'will save the file to ./data/train/{mode}_landmark.txt')

    whole_list = os.listdir(f'./data/test/{mode}_set/test_pairs')
    # sort whole_list in number order
    whole_list.sort( key = lambda s: int( s.split('_')[2] ) )
    # transpose so that we can directly return self.pair_list[index]
    # pair_list[i] = [filename1, filename2]
    pair_list = np.array([whole_list[::2], whole_list[1::2]]).T

    detector = MTCNN()
    face_list = []
    for filenames in tqdm(pair_list):
        img1 = cv2.cvtColor(cv2.imread(os.path.join(f'./data/test/{mode}_set/test_pairs', filenames[0])), cv2.COLOR_BGR2RGB )
        img2 = cv2.cvtColor(cv2.imread(os.path.join(f'./data/test/{mode}_set/test_pairs', filenames[1])), cv2.COLOR_BGR2RGB )

        face_list.append([
            findBestFace(detector.detect_faces(img1)),
            findBestFace(detector.detect_faces(img2))
        ])
        del img1
        del img2

    # get flag (testing answer)
    with open(f'./data/test/{mode}_set/labels.txt') as f:
        flag_list = f.readlines()
        flag_list = list(map(lambda s: int(s[:-1]), flag_list))

    with open(f'./data/test/{mode}_landmark.txt', 'w') as f:
        for i in range(len(face_list)):
            s = f'{flag_list[i]}\t' + \
                f'{pair_list[i][0]}\t' + \
                f'{face_list[i][0]["left_eye"][0]}\t{face_list[i][0]["left_eye"][1]}\t' + \
                f'{face_list[i][0]["right_eye"][0]}\t{face_list[i][0]["right_eye"][1]}\t' + \
                f'{face_list[i][0]["nose"][0]}\t{face_list[i][0]["nose"][1]}\t' + \
                f'{face_list[i][0]["mouth_left"][0]}\t{face_list[i][0]["mouth_left"][1]}\t' + \
                f'{face_list[i][0]["mouth_right"][0]}\t{face_list[i][0]["mouth_right"][1]}\t' + \
                f'{pair_list[i][1]}\t' + \
                f'{face_list[i][1]["left_eye"][0]}\t{face_list[i][1]["left_eye"][1]}\t' + \
                f'{face_list[i][1]["right_eye"][0]}\t{face_list[i][1]["right_eye"][1]}\t' + \
                f'{face_list[i][1]["nose"][0]}\t{face_list[i][1]["nose"][1]}\t' + \
                f'{face_list[i][1]["mouth_left"][0]}\t{face_list[i][1]["mouth_left"][1]}\t' + \
                f'{face_list[i][1]["mouth_right"][0]}\t{face_list[i][1]["mouth_right"][1]}\n'
            f.write(s)

def main():
    if args.mode == 'train':
        genTrain()
    elif args.mode == 'close' or args.mode == 'closed':
        genTest('closed')
    elif args.mode == 'open':
        genTest('open')
    else:
        print('[genLandmark] generating mode is incorrect!')

if __name__ == '__main__':
    main()
