echo "-- Processing val set..."
python data_argo/run_preprocess.py --mode val \
  --data_dir ~/argoverse-dataset/argoverse/val/data/ \
  --save_dir data_argo/features/ \
  --small
  # --debug --viz

echo "-- Processing train set..."
python data_argo/run_preprocess.py --mode train \
  --data_dir ~/argoverse-dataset/argoverse/train/data \
  --save_dir data_argo/features/ \
  --small

