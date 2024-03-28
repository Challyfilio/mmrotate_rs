#!/bin/bash
#img_list='P0500 P0628 P0680 P0681 P1511 P2645'
#img_list='P0011 P0029 P0628 P0680 P1511 P0871'
#img_list='P2212 P2515 P2573 P2643 P2694 P2782'
img_list='P2694_15'

#MODE=$1

cd ..

#for IMG in ${img_list}; do
#  python demo/image_demo.py \
#   demo/${IMG}.png \
#   configs/oriented_rcnn/oriented_rcnn_clip_bbox_head_1x_le90.py \
#   work_dir/rsp-r50dc_clip_bbox_head_dota_ms_trainval/latest.pth \
#   --out-file ${IMG}_rsp-r50dc_clip.jpg
#done

#for IMG in ${img_list}; do
#  python demo/image_demo.py \
#   demo/${IMG}.png \
#   configs/oriented_rcnn/oriented_rcnn_clip_bbox_head_1x_le90.py \
#   work_dir/oriented_rcnn_rsp-r50_clip_bbox_head_dota_ss_trainval/latest.pth \
#   --out-file ${IMG}_rsp-r50_clip.jpg
#done

for IMG in ${img_list}; do
  python demo/image_demo.py \
   demo/${IMG}.png \
   configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py \
   /workspace/pycharm_project/mmrotate/pretrain/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
   --out-file ${IMG}_orcnn-r50.jpg
done