device=0

LOG=${save_dir}"res.log"
echo ${LOG}


data_root=./dataset #<Set_YOUR_DATASET_DIR>
n_ctx=12
vl_reduction=4
pq_mid_dim=128


# test on VisA dataset, model trained on MVTec dataset
train_dataset=mvtec
test_dataset=visa
for shot in 0 1 2 4
do
    if [ ${shot} -eq 0 ]; then
        seeds="10"
    else
        seeds="10 20 30"
    fi

    for seed in ${seeds} 
    do 
        base_dir=${n_ctx}_${vl_reduction}_${pq_mid_dim}_train_on_${train_dataset}_3adapters_batch8
        
        
        save_dir=./results/${base_dir}
        model_dir=./adaptclip_checkpoints/${base_dir}

        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset ${test_dataset}  --test_data_path ${data_root}/${test_dataset} \
        --seed ${seed} \
        --k_shots ${shot} \
        --checkpoint_path ${model_dir}/epoch_15.pth \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 --image_size 518  --batch_size 8  \
        --n_ctx ${n_ctx}  --vl_reduction ${vl_reduction} --pq_mid_dim ${pq_mid_dim} \
        --visual_learner --textual_learner --pq_learner  --pq_context 
    wait
    done
done


# test on MVTec dataset, model trained on VisA dataset
train_dataset=mvtec
test_dataset=visa
for shot in 0 1 2 4
do
    if [ ${shot} -eq 0 ]; then
        seeds="10"
    else
        seeds="10 20 30"
    fi

    for seed in ${seeds} 
    do 
        base_dir=${n_ctx}_${vl_reduction}_${pq_mid_dim}_train_on_${train_dataset}_3adapters_batch8
        
        save_dir=./results/${base_dir}
        model_dir=./adaptclip_checkpoints/${base_dir}

        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset ${test_dataset}  --test_data_path ${data_root}/${test_dataset} \
        --seed ${seed} \
        --k_shots ${shot} \
        --checkpoint_path ${model_dir}/epoch_15.pth \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 --image_size 518  --batch_size 8  \
        --n_ctx ${n_ctx}  --vl_reduction ${vl_reduction} --pq_mid_dim ${pq_mid_dim} \
        --visual_learner --textual_learner --pq_learner  --pq_context 
    wait
    done
done
