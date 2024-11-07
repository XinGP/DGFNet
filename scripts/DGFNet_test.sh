CUDA_VISIBLE_DEVICES=0 python test.py \
  --features_dir data_argo/features/ \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --use_cuda \
  --adv_cfg_path config.DGFNet_cfg \
  --model_path saved_models/20240707-211851_DGFNet_best.tar
