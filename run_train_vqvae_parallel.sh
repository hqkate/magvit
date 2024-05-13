# export MS_ENABLE_REF_MODE=1 # NEEDED for 910B + MS2.2 0907 version for saving checkpoint correctly
export MS_ASCEND_CHECK_OVERFLOW_MODE=1 # for ms+910B, check overflow
#export INF_NAN_MODE_ENABLE=1 # For pytorch+npu, recommend to enable it for mixed precision training for 910B. it determines how overflow is detected

task_name=train_vqvae_ucf101_8p
output_path=outputs

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
# uncomment this following line for caching and loading the compiled graph, which is saved in ${output_path}/${task_name}_cache
# export MS_COMPILER_CACHE_ENABLE=1
mkdir -p ${output_path:?}/${task_name:?}_cache
export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}_cache

# Parallel config
num_devices=8
rank_table_file=/disk3/katekong/hccl/hccl_8p_01234567_127.0.0.1.json
# CANDIDATE_DEVICE=(0 1 2 3 4 5 6 7)

# ascend config
#export GLOG_v=3
# export HCCL_CONNECT_TIMEOUT=6000 # the real error info in modelarts can be blocked by the timeout error if this value is larger than HCCL_EXEC_TIMEOUT!
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=0

ulimit -u unlimited
ulimit -SHn 65535
export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
RANK_TABLE_FILE=$rank_table_file
export RANK_TABLE_FILE=${RANK_TABLE_FILE}
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

# remove files
output_dir=$output_path/$task_name
cp $0 $output_dir/.


for((i=0; i<${RANK_SIZE}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    mkdir -p ${output_dir:?}//rank_$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    nohup python train_vqvae.py \
        --use_parallel True \
        --use_discriminator True \
        --use_ema True \
        --dataset_name video \
        --data_path /disk3/katekong/magvit/datasets/ucf101/middlebatch/ \
        --num_frames 16 \
        --crop_size 128 \
        --num_parallel_workers 8 \
        --drop_overflow_update False \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --base_learning_rate 1.0e-04 \
        --dtype fp32 \
        --mode 0 \
        > $output_dir/rank_$i/train.log 2>&1 &
done