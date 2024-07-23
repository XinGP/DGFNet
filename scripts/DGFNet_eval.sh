CUDA_VISIBLE_DEVICES=0 python evaluation.py \
  --features_dir data_argo/features/ \
  --train_batch_size 16 \
  --val_batch_size 16 \
  --use_cuda \
  --adv_cfg_path config.DGFNet_cfg \
  --model_path saved_models/20240722-163842_DGFNet_epoch2.tar