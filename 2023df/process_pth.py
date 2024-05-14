# Copyright (c) 2023 ✨Challyfilio✨
# 处理pth中的键
import torch
from loguru import logger

pretrained_A = '/workspace/pycharm_project/mmrotate/pretrain/resnet50-19c8e357.pth'  # A
pretrained_B = '/workspace/pycharm_project/mmrotate/pretrain/rsp-resnet-50.pth'  # bs B

pretrained_c = '/workspace/pycharm_project/mmrotate/work_dir/test_exp3/epoch_6.pth'

if __name__ == '__main__':
    pretrained_dict_A = torch.load(pretrained_A)
    print(pretrained_dict_A.keys())
    # print(pretrained_dict_A['state_dict'].keys())
    pretrained_dict_B = torch.load(pretrained_B)
    print(pretrained_dict_B.keys())
    # print(pretrained_dict_B['model'].keys())
    exit()
    # for keys in pretrained_dict_A['state_dict'].keys():
    #     if 'backbone.' or 'neck.' or 'rpn_head.' in keys:
    #         pass
    #     else:
    #         del pretrained_dict_A['state_dict'][keys]
    # torch.save(pretrained_dict_A, 'orcn-rsp-r50-fpn-rpn-dota.pth')
    # logger.success('finish')
    # exit()

    print(pretrained_dict_A['conv1.weight'])
    print(pretrained_dict_B['model']['conv1.weight'])

    for a_key in pretrained_dict_A.keys():
        pretrained_dict_A[a_key] = pretrained_dict_B['model'][a_key]

    print(pretrained_dict_A['conv1.weight'])
    torch.save(pretrained_dict_A, "rsp-resnet-50.pth")
    logger.success('finish')
    exit()

    count = 0
    for a_key in list(pretrained_dict_A['state_dict'].keys()):
        for b_key in list(pretrained_dict_B['state_dict'].keys()):
            if a_key == b_key:
                if pretrained_dict_B['state_dict'][b_key].shape == pretrained_dict_A['state_dict'][a_key].shape:
                    # logger.info(a_key)
                    pretrained_dict_B['state_dict'][b_key] = pretrained_dict_A['state_dict'][a_key]
                    count += 1
                else:
                    pass
    print(count)
    # print(pretrained_dict_B.keys())
    # print(len(pretrained_dict_B['state_dict'].keys()))
    torch.save(pretrained_dict_B, "my_model.pth")
    logger.success('finish')
