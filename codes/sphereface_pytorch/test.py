from net_sphere import sphere20a
import argparse
from lfw_eval import alignment
import numpy as np
import os
import torch
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest='mode', type=str)
parser.add_argument('--path', dest='model_path', type=str, default='./models/sphere20a_29.pth')
parser.add_argument('--feature', dest='use_feature', action='store_true')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def loadNet(path=args.model_path):
    net = sphere20a()
    net.load_state_dict(torch.load(path))
    net.to(device)
    net.eval()
    return net

def testClose(use_feature=False):
    """
    in close mode, directly get the face id and compare
    (without getting feature and compare distance)
    """
    # load model
    net = loadNet(args.model_path)
    if use_feature:
        net.feature = True
        threshold = 0.25

    predicts = []
    ground_truth = []

    with open('../data/test/closed_landmark.txt', 'r') as f:
        landmark_lines = f.readlines()

    landmark_lines = np.array([ l[:-1].split('\t') for l in landmark_lines ])

    for i, line in enumerate(tqdm(landmark_lines)):
        same = int(line[0])
        ground_truth.append(same)
        file1 = line[1]
        landmark1 = [ int(x) for x in line[2:12] ]
        file2 = line[12]
        landmark2 = [ int(x) for x in line[13:] ]

        with open(f'../data/test/closed_set/test_pairs/{file1}', 'rb') as f:
            img1 = alignment(cv2.imdecode(np.frombuffer(f.read(),np.uint8),1),landmark1)
        with open(f'../data/test/closed_set/test_pairs/{file2}', 'rb') as f:
            img2 = alignment(cv2.imdecode(np.frombuffer(f.read(),np.uint8),1),landmark2)

        img_list = [img1, img2]
        for i, img in enumerate(img_list):
            img_list[i] = img.transpose(2, 0, 1).reshape((1, 3, 112, 96))
            img_list[i] = (img_list[i] - 127.5) / 128.0

        outputs = [
            net(torch.from_numpy(img_list[0]).float().to(device)),
            net(torch.from_numpy(img_list[1]).float().to(device))
        ]
        if not use_feature:
            _, pred1 = torch.max(outputs[0][0], 1)
            _, pred2 = torch.max(outputs[1][0], 1)
            result = 1 if pred1 == pred2 else 0
        else:
            v1 = outputs[0][0]
            v2 = outputs[1][0]
            similarity = v1.dot(v2) / (v1.norm() * v2.norm())
            # quantize by threshold
            result = 1 if similarity >= threshold else 0
        predicts.append(result)

    correct = np.sum([ predicts[i] == ground_truth[i] for i in range(len(predicts)) ])
    acc = correct / len(predicts)

    print(f'mode: close\nAccurcy: {acc:.5f}, correct/all: {correct}/{len(predicts)}')

def testOpen():
    """
    in open mode, we compare two faces with their feature
    """
    # load net
    net = loadNet(args.model_path)
    # output feature
    net.feature = True

    predicts = []
    ground_truth = []
    threshold = 0.25

    with open('../data/test/open_landmark.txt', 'r') as f:
        landmark_lines = f.readlines()

    landmark_lines = np.array([ l[:-1].split('\t') for l in landmark_lines ])

    for i, line in enumerate(tqdm(landmark_lines)):
        same = int(line[0])
        ground_truth.append(same)
        file1 = line[1]
        landmark1 = [ int(x) for x in line[2:12] ]
        file2 = line[12]
        landmark2 = [ int(x) for x in line[13:] ]

        with open(f'../data/test/open_set/test_pairs/{file1}', 'rb') as f:
            img1 = alignment(cv2.imdecode(np.frombuffer(f.read(),np.uint8),1),landmark1)
        with open(f'../data/test/open_set/test_pairs/{file2}', 'rb') as f:
            img2 = alignment(cv2.imdecode(np.frombuffer(f.read(),np.uint8),1),landmark2)

        img_list = [img1, img2]
        for i, img in enumerate(img_list):
            img_list[i] = img.transpose(2, 0, 1).reshape((1, 3, 112, 96))
            img_list[i] = (img_list[i] - 127.5) / 128.0

        outputs = [
            net(torch.from_numpy(img_list[0]).float().to(device)),
            net(torch.from_numpy(img_list[1]).float().to(device))
        ]
        v1 = outputs[0][0]
        v2 = outputs[1][0]
        similarity = v1.dot(v2) / (v1.norm() * v2.norm())
        # quantize by threshold
        result = 1 if similarity >= threshold else 0
        predicts.append(result)

    correct = np.sum([ predicts[i] == ground_truth[i] for i in range(len(predicts)) ])
    acc = correct / len(predicts)

    print(f'mode: open\nthreshold: {threshold}\nAccurcy: {acc:.5f}, correct/all: {correct}/{len(predicts)}')


def main():

    if args.mode == 'close':
        testClose(use_feature=args.use_feature)
    elif args.mode == 'open':
        testOpen()
    else:
        print("[sphereface] please specify your test mode!")

if __name__ == '__main__':
    main()
