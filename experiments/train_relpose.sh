# Our standard RelPose model
CUDA_VISIBLE_DEVICES=7 python ../train.py --model_name mono+relpose \
    --pose_model_type relpose \
    --models_to_load encoder depth \
    --load_weights_folder ../../checkpoints \
    --data_path ../../data/kitti_data \
    --log_dir ../logs/