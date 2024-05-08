#!/bin/bash

gpu_id=0
gpu_list=(3)
batch_size_list=(128 256)
optimizer_lr_list=(0.1 0.2)
optimizer_weight_decay_list=(0.0001 0.0005)
scheduler_gamma=(0.5 0.4 0.3)
scheduler_milestones_offset_list=(0 5)

for batsz in ${batch_size_list[@]}; do
  for lr in ${optimizer_lr_list[@]}; do
    for decay in ${optimizer_weight_decay_list[@]}; do
      for gamma in ${scheduler_gamma[@]}; do
        for offset in ${scheduler_milestones_offset_list[@]}; do
          python3 main.py \
            --record_directory "/data1/ljh/experiments_tiny_imagenet_sgd_256" \
            --dataset_root "/data1/ljh/datasets" \
            --dataset_name "TinyImageNet" \
            --router_filter_number 128 \
            --transformer_filter_number 256 \
            --datloader1_train_batsz ${batsz} \
            --datloader1_valid_batsz 100 \
            --datloader1_test_batsz 100 \
            --optimizer1_epochs 100 \
            --optimizer1_sgd_or_adamw \
            --optimizer1_lr ${lr} \
            --optimizer1_weight_decay ${decay} \
            --scheduler1_milestones $[25 - ${offset}] $[45 - ${offset}] $[65 - ${offset}] $[85 - ${offset}] \
            --scheduler1_gamma ${gamma} \
            --datloader2_train_batsz ${batsz} \
            --datloader2_valid_batsz 100 \
            --datloader2_test_batsz 100 \
            --optimizer2_epochs 100 \
            --optimizer2_sgd_or_adamw \
            --optimizer2_lr ${lr} \
            --optimizer2_weight_decay 0.0005 \
            --scheduler2_milestones $[25 - ${offset}] $[45 - ${offset}] $[65 - ${offset}] $[85 - ${offset}] \
            --scheduler2_gamma ${gamma} \
            --gpu ${gpu_list[gpu_id]}
          if [[ $? != 0 ]]; then
            gpu_id=$[(${gpu_id} + 1) % ${#gpu_list[*]}]
          fi
          sleep 3
        done
      done
    done
  done
done
