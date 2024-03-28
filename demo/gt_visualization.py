# Copyright (c) 2024 ✨Challyfilio✨
'''
Created on Mar 5, 2024
可视化ground-truth
'''
import cv2
import os
import numpy as np

PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
           (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
           (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
           (255, 255, 0), (147, 116, 116), (0, 0, 255)]

CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter')


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def zhongdian(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def gt_vis(img_path, txt_path):
    bsn = get_file_basename(img_path)
    count = 0
    with open(txt_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
            if os.path.exists('gt_' + bsn + '.jpg'):
                img = cv2.imread('gt_' + bsn + '.jpg')
            else:
                img = cv2.imread(img_path)
            data = []
            for i in range(0, 8):
                position = l.split(' ')[i]
                data.append(float(position))
            cls = l.split(' ')[8]

            cnt = np.array([
                [[data[0], data[1]]],
                [[data[2], data[3]]],
                [[data[4], data[5]]],
                [[data[6], data[7]]]
            ], dtype=np.float32)
            # print("shape of cnt: {}".format(cnt.shape))
            rect = cv2.minAreaRect(cnt)
            # print("rect: {}".format(rect))

            # the order of the box points: bottom left, top left, top right,
            # bottom right
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(img, [box], 0, PALETTE[CLASSES.index(cls)][::-1], 2)

            font = cv2.FONT_HERSHEY_SIMPLEX

            (w, h), _ = cv2.getTextSize(cls, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            c_x, c_y = zhongdian(data[0], data[1], data[4], data[5])
            cv2.rectangle(img, (c_x, c_y), (c_x + w, c_y - h), (0, 0, 0), -1)

            cv2.putText(img,
                        cls,
                        (c_x, c_y),
                        font,
                        1,
                        (255, 255, 255),
                        2)
            # cv2.putText(img, label, (int(eval(row[3])), int(eval(row[4]))), font, 1, (0, 0, 255), 1)
            cv2.imwrite('gt_' + bsn + '.jpg', img)
            count += 1
        print(count)


def tuple_reverse(a_tuple):
    return a_tuple[::-1]


if __name__ == "__main__":
    img_path = '/workspace/pycharm_project/mmrotate/2023df/P2150.png'
    txt_path = '/workspace/pycharm_project/mmrotate/2023df/P2150.txt'
    gt_vis(img_path, txt_path)

# cv2.rectangle(image, (x, x), (x + w, y + h), (0,0,0), -1)
