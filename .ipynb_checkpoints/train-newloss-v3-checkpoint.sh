device=6

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
FEATURES_LIST=(9 12 15 18 21 24)
num_prototypes=64
pretrained_dataset=mvtec-clinic
pretrained_data_path='data/mvtec-clinic'
#loss 超参数  9 12 15 18 21 24
dice_abn=1
focal_1=5
focal_2=95
focal_gamma=3
prompt_num=12

for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${num_prototypes}_preby_${pretrained_dataset}-${prompt_num}prompt-newloss-v3
        save_dir=./save_ckpt/${base_dir}
        CUDA_VISIBLE_DEVICES=${device} python train-newloss-v3.py --dataset ${pretrained_dataset} --train_data_path ${pretrained_data_path} \
        --save_path ${save_dir} \
        --features_list "${FEATURES_LIST[@]}" \
        --batch_size 48 --num_prototypes ${num_prototypes} \
        --epoch 80 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --dice_abn ${dice_abn} --focal_1 ${focal_1} --focal_2 ${focal_2} --focal_gamma ${focal_gamma} --prompt_num ${prompt_num}
    wait
    done
done