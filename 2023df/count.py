# Copyright (c) 2024 ✨Challyfilio✨
# 2024/2/21
# 统计数据集
import os
from tqdm import tqdm


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


if __name__ == '__main__':
    cls_list = []
    cls_dict = {}
    data_root = '/workspace/pycharm_project/mmrotate/data/val/'
    imgfiles = os.listdir(data_root + 'images')
    for imgfile in tqdm(imgfiles):
        img_path = os.path.join(data_root + 'images', imgfile)
        count = 0
        basename = get_file_basename(imgfile)
        label_txt = data_root + 'labelTxt-v1.0/' + basename + '.txt'
        with open(label_txt, "r", encoding='utf-8') as f:
            lines = f.readlines()[2:]
            for l in lines:
                alist = []
                for i in range(0, 8):
                    alist.append(float(l.split(' ')[i]))
                cls = l.split(' ')[8].replace('\n', '')
                if cls not in cls_list:
                    cls_list.append(cls)
                    cls_dict.update({cls: 1})
                else:
                    cls_dict[cls] += 1
    print(len(cls_list))
    print(len(cls_dict))
    print(cls_dict)
