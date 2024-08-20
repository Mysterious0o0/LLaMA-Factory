#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file orderTrain/multi_gpu_config.yaml \
    src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /home/songfucheng/model/yi \
    --dataset order_data \
    --dataset_dir orderTrain \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \ 
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target W_pack \
    --output_dir orderTrain/yi/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
	--max_grad_norm 1.0 \
    --logging_steps 10 \
    --warmup_steps 100 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --load_best_model_at_end False \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 10000 \
    --val_size 0.1 \
    --ddp_timeout 18000000 \
    --plot_loss True \
    --fp16 True \
	--report_to wandb > orderTrain/task1.log 2>&1 &



#!/bin/bash

nohup CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /home/songfucheng/model/baichuan2 \
    --dataset order_data \
    --dataset_dir orderTrain \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \ 
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target W_pack \
    --output_dir orderTrain/baichuan2/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
	--max_grad_norm 1.0 \
    --logging_steps 10 \
    --warmup_steps 100 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --load_best_model_at_end False \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 10000 \
    --val_size 0.1 \
    --ddp_timeout 18000000 \
    --plot_loss True \
    --fp16 True \
	--report_to wandb > orderTrain/task1.log 2>&1 &



#!/bin/bash
nohup deepspeed --num_gpus 4 src/train_bash.py \
    --deepspeed orderTrain/multi_ds_config.json \
    --ddp_timeout 180000000 \
    --stage sft \
    --do_train \
    --model_name_or_path /home/songfucheng/model/baichuan2 \
    --dataset order_data \
    --dataset_dir orderTrain \
    --template baichuan2 \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target W_pack \
    --output_dir orderTrain/baichuan2/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --warmup_steps 100 \
    --save_steps 100 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end False \
    --learning_rate 5e-5 \
    --num_train_epochs 6.0 \
    --max_samples 10000 \
    --val_size 0.1 \
    --ddp_timeout 18000000 \
    --plot_loss True \
    --fp16 True \
    --report_to wandb > orderTrain/task1.log 2>&1 &

nohup torchrun --nproc_per_node 4 src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/songfucheng/model/baichuan2 \
    --do_train \
    --dataset order_data \
    --dataset_dir orderTrain \
    --template baichuan2 \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target W_pack \
    --output_dir orderTrain/baichuan2/lora1/sft \
    --overwrite_cache \
    --per_device_train_batch_size 4 \ 
    --per_device_eval_batch_size 4 \ 
    --gradient_accumulation_steps 8 \ 
    --preprocessing_num_workers 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 1e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2.0 \
    --dev_ratio 0.01 \
    --ddp_timeout 18000000 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16 True \
    --report_to wandb > orderTrain/task1.log 2>&1 &



# 导出模型
CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
    --model_name_or_path /home/songfucheng/model/baichuan2 \
    --adapter_name_or_path orderTrain/baichuan2/lora/sft/checkpoint-250 \
    --template default \
    --finetuning_type lora \
    --export_dir orderTrain/baichuan2/lora/ds \
    --export_size 2 \
    --export_legacy_format False


# 浏览器推理
CUDA_VISIBLE_DEVICES=0,1 python src/web_demo.py \
    --model_name_or_path /home/songfucheng/model/baichuan2 \
    --adapter_name_or_path orderTrain/baichuan2/lora/sft/checkpoint-2200 \
    --template default \
    --finetuning_type lora




deepspeed.launcher.launch \
--world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 \
--master_addr=127.0.0.1 \
--master_port=29500 \
--enable_each_rank_log=None src/train_bash.py \
--deepspeed orderTrain/multi_ds_config.json \
--ddp_timeout 180000000 \
--stage sft \
--do_train \
--model_name_or_path /home/songfucheng/model/baichuan2 \
--dataset order_data \
--dataset_dir orderTrain \
--template default \
--finetuning_type lora \
--lora_rank 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target W_pack \
--output_dir orderTrain/baichuan2/lora/sft2 \
--overwrite_cache \
--overwrite_output_dir \
--cutoff_len 2048 \
--preprocessing_num_workers 8 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--max_grad_norm 1.0 \
--logging_steps 10 \
--warmup_steps 100 \
--save_steps 50 \
--eval_steps 50 \
--evaluation_strategy steps \
--load_best_model_at_end False \
--learning_rate 5e-5 \
--num_train_epochs 6.0 \
--max_samples 10000 \
--val_size 0.1 \
--ddp_timeout 18000000 \
--plot_loss True \
--fp16 True \
--report_to wandb




#!/bin/bash
nohup deepspeed --num_gpus 4 src/train_bash.py \
    --deepspeed orderTrain/multi_ds_config.json \
    --ddp_timeout 180000000 \
    --stage sft \
    --do_train \
    --model_name_or_path /home/songfucheng/model/yi \
    --dataset order_data \
    --dataset_dir orderTrain \
    --template yi \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --output_dir orderTrain/yi/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --model_max_length 100 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --warmup_steps 100 \
    --save_steps 100 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end False \
    --learning_rate 5e-5 \
    --num_train_epochs 6.0 \
    --max_samples 10000 \
    --val_size 0.1 \
    --ddp_timeout 18000000 \
    --plot_loss True \
    --fp16 True \
    --run_name yi-9b \
    --report_to wandb > orderTrain/task1.log 2>&1 &


















CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup deepspeed --master_port 25678 src/train_bash.py \
--deepspeed orderTrain/multi_ds_config.json \
--ddp_timeout 180000000 \
--stage sft \
--do_train \
--model_name_or_path /home/lc/model/Mixtral-8x7B-Instruct-v0.1 \
--dataset order_data \
--dataset_dir orderTrain \
--template mistral \
--finetuning_type lora \
--lora_rank 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target all \
--output_dir orderTrain/Mixtral/lora/sft2 \
--overwrite_cache \
--overwrite_output_dir \
--cutoff_len 6000 \
--preprocessing_num_workers 8 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--max_grad_norm 1.0 \
--logging_steps 10 \
--warmup_steps 100 \
--save_steps 100 \
--eval_steps 50 \
--evaluation_strategy steps \
--load_best_model_at_end False \
--learning_rate 5e-5 \
--num_train_epochs 6.0 \
--max_samples 10000 \
--val_size 0.1 \
--plot_loss True \
--fp16 True \
--run_name Mixtral-8x7b \
--report_to wandb > orderTrain/Mixtral.log 2>&1 &


CUDA_VISIBLE_DEVICES=3,4,5,6 和 --num_gpus 4选一个就行不能都写  master_port:通信端口号


CUDA_VISIBLE_DEVICES=0,1 python src/web_demo.py \
    --model_name_or_path /home/songfucheng/model/yi \
    --adapter_name_or_path orderTrain2/yi/lora/sft/checkpoint-2200 \
    --template yi \
    --finetuning_type lora \
    --quantization_bit 4 \
    --quantization_device_map auto


# 导出模型
CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
    --model_name_or_path /home/songfucheng/model/yi \
    --adapter_name_or_path orderTrain2/yi/lora/sft/checkpoint-2200 \
    --template yi \
    --finetuning_type lora \
    --export_dir orderTrain2/yi/lora/ds \
    --export_quantization_bit 4 \
    --export_quantization_dataset ./data/c4_demo.json \
    --export_size 2 \
    --export_legacy_format False

