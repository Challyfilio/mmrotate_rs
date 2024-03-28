# Copyright (c) 2023 ✨Challyfilio✨
"""
可视化检测结果，用于系统后端
"""
# 导入os模块
import os

cfg_baseline = dict(
    outfile='_baseline',
    config_path='/workspace/pycharm_project/mmrotate/pretrain/oriented_rcnn_r50_fpn_1x_dota_le90.py',
    checkpoint_path='/workspace/pycharm_project/mmrotate/pretrain/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth',
)

cfg_exp1 = dict(
    outfile='_exp1',
    config_path='/workspace/pycharm_project/mmrotate/configs/oriented_rcnn/exp1_oriented_rcnn_r50dc_ffpn_1x_dota_le90.py',
    checkpoint_path='/workspace/pycharm_project/mmrotate/inference/oriented_rcnn_r50_dcn_pafpnwo_adapter_ce_dota_ms_traintest0.8/epoch_12.pth',
)

cfg_exp2 = dict(
    outfile='_exp2',
    config_path='/workspace/pycharm_project/mmrotate/configs/oriented_rcnn/exp2_oriented_rcnn_r50dc_clip_bbox_head_1x_dota_le90.py',
    checkpoint_path='/workspace/pycharm_project/mmrotate/inference/rsp-r50dc_clip_bbox_head_dota_ms_trainval/epoch_12.pth',
)

cfg_exp3 = dict(
    outfile='_exp3',
    config_path='/workspace/pycharm_project/mmrotate/configs/oriented_rcnn/exp3_oriented_rcnn_r50_prototype_bbox_head_1x_dota_le90.py',
    checkpoint_path='/workspace/pycharm_project/mmrotate/inference/r50_fpn_ce_dota_ss_5000shot_test/epoch_36.pth',
)

'''
python demo/image_demo.py \
    demo/demo.jpg oriented_rcnn_r50_fpn_1x_dota_le90.py \
    oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
    --out-file result.jpg
'''


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def vis_rotated_result(image_path, config):
    outfile_path = ('/workspace/pycharm_project/mmrotate/demo/result_' + get_file_basename(image_path) +
                    config['outfile'] + '.jpg')
    config_path = config['config_path']
    checkpoint_path = config['checkpoint_path']
    # 拼接命令字符串
    cmd = 'python ../demo/image_demo.py' + ' ' + image_path + ' ' + config_path + ' ' + checkpoint_path + ' --out-file ' + outfile_path
    # 执行命令字符串
    os.system(cmd)


if __name__ == '__main__':
    image_path = '/workspace/pycharm_project/mmrotate/demo/21406.jpg'
    # vis_rotated_result(image_path, cfg_baseline)
    # vis_rotated_result(image_path, cfg_exp1)
    # vis_rotated_result(image_path, cfg_exp2)
    vis_rotated_result(image_path, cfg_exp3)
