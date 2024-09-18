nohup CUDA_VISIBLE_DEVICES=2,3 deepspeed src/train.py \
    --deepspeed orderTrain/multi_ds_config.json \
    --master_port 7777 \
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



CUDA_VISIBLE_DEVICES=2,3 nohup torchrun  --nproc_per_node=2 --master_port 16667 src/train.py \
  --model_name_or_path /home/data/cipher/model/qwen/qwen1.5-72B-Chat-GPTQ-Int4 \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_target all \
  --lora_dropout 0.1 \
  --dataset talk_data \
  --dataset_dir dataProcess \
  --template qwen \
  --cutoff_len 4096 \
  --max_samples 10000 \
  --overwrite_cache true \
  --preprocessing_num_workers 16 \
  --output_dir output/qwen/Qwen1.5-72B-Chat-GPTQ-Int4/lora/sft3 \
  --logging_steps 10 \
  --save_steps 50 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5.0e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --val_size 0.1 \
  --per_device_eval_batch_size 4 \
  --eval_strategy steps \
  --eval_steps 20 \
  --report_to wandb \
  --run_name qwen1.5-chat-int4 > dataProcess/qwen1.5-chat-int4.log 2>&1 &




### model
CUDA_VISIBLE_DEVICES=1,2,3 nohup torchrun  --nproc_per_node=3 src/train.py \
  --model_name_or_path /home/data/cipher/model/qwen/Qwen1.5-72B-Chat \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_target all \
  --lora_dropout 0.1 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset talk_data \
  --dataset_dir dataProcess \
  --template qwen \
  --cutoff_len 4096 \
  --max_samples 10000 \
  --overwrite_cache true \
  --preprocessing_num_workers 8 \
  --output_dir output/qwen/Qwen1.5-72B-Chat/lora/sft3 \
  --logging_steps 10 \
  --save_steps 20 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5.0e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --val_size 0.1 \
  --per_device_eval_batch_size 2 \
  --eval_strategy steps \
  --eval_steps 10 \
  --report_to wandb \
  --run_name qwen1.5-chat > dataProcess/qwen1.5-chat3.log 2>&1 &


CUDA_VISIBLE_DEVICES=2,3 nohup torchrun --nproc_per_node=2 --master_port 16666 src/train.py \
  --model_name_or_path /home/data/cipher/model/deepseek-ai/DeepSeek-V2-Lite-Chat \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_target all \
  --lora_dropout 0.1 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset talk_data \
  --dataset_dir dataProcess \
  --template deepseek \
  --cutoff_len 4096 \
  --max_samples 10000 \
  --overwrite_cache true \
  --preprocessing_num_workers 16 \
  --output_dir output/deepseek/DeepSeek-V2-Lite-Chat/lora/sft \
  --logging_steps 20 \
  --save_steps 40 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5.0e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --val_size 0.01 \
  --per_device_eval_batch_size 4 \
  --eval_strategy steps \
  --eval_steps 200 \
  --report_to wandb \
  --run_name deepseekv2-chat > dataProcess/DeepSeekV2-chat.log 2>&1 &


# 5555
CUDA_VISIBLE_DEVICES=1,2,3 python src/web_demo.py \
    --model_name_or_path /home/data/cipher/model/qwen/Qwen1.5-72B-Chat \
    --adapter_name_or_path output/qwen/Qwen1.5-72B-Chat/lora/sft3/checkpoint-100 \
    --template qwen \
    --finetuning_type lora

# 6666
CUDA_VISIBLE_DEVICES=0,1 python src/web_demo1.py \
    --model_name_or_path /home/data/cipher/model/qwen/qwen1.5-72B-Chat-GPTQ-Int4 \
    --adapter_name_or_path output/qwen/Qwen1.5-72B-Chat-GPTQ-Int4/lora/sft3/checkpoint-100 \
    --template qwen \
    --finetuning_type lora
