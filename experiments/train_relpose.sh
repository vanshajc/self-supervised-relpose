# Our Monodepth2 baseline on KITTI-Odom
# CUDA_VISIBLE_DEVICES=6 python ../train.py --model_name mono_kitti_odom \
#     --split odom --dataset kitti_odom \
#     --data_path ../../data/kitti_odom \
#     --log_dir ../logs/

# Our standard RelPose model on KITTI-Odom
CUDA_VISIBLE_DEVICES=7 python ../train.py --model_name mono_unfrozen+relpose_kitti_odom \
    --pose_model_type relpose \
    --split odom --dataset kitti_odom \
    --data_path ../../data/kitti_odom \
    --log_dir ../logs/

# Our standard RelPose model on KITTI-RAW
# CUDA_VISIBLE_DEVICES=7 python ../train.py --model_name mono_unfrozen+relpose_kitti_raw \
#     --pose_model_type relpose \
#     --data_path ../../data/kitti_data \
#     --log_dir ../logs/
# RelPose with pre-trained depth and encoder
# CUDA_VISIBLE_DEVICES=7 python ../train.py --model_name mono_unfrozen+relpose_kitti_raw \
#     --pose_model_type relpose \
#     --data_path ../../data/kitti_data \
#     --log_dir ../logs/ \
#     --models_to_load encoder depth \
#     --load_weights_folder ../../checkpoints