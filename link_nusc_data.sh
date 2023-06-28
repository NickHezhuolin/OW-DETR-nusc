#!/bin/bash

# 首先定义源目录和目标目录
src_dirs=("/data/dataset2tssd/nuscenes/samples/CAM_BACK" "/data/dataset2tssd/nuscenes/samples/CAM_BACK_LEFT" "/data/dataset2tssd/nuscenes/samples/CAM_BACK_RIGHT" "/data/dataset2tssd/nuscenes/samples/CAM_FRONT" "/data/dataset2tssd/nuscenes/samples/CAM_FRONT_LEFT" "/data/dataset2tssd/nuscenes/samples/CAM_FRONT_RIGHT")
dst_dir="/home/hez4sgh/1_workspace/OW-DETR-nusc/data/OWDETR/Nuscenes/JEPGImage"

# 创建目标目录，如果它还不存在的话
mkdir -p $dst_dir

# 对每一个源目录进行操作
for src_dir in ${src_dirs[@]}; do
  # 找到源目录中的所有.jpg文件（或者你需要的其他类型的图片文件）
  for src_file in $(find $src_dir -name '*.jpg'); do
    # 使用源文件的完整路径和文件名创建目标文件的路径
    dst_file="$dst_dir/$(basename $src_file)"
    # 创建软链接
    ln -s $src_file $dst_file
  done
done
