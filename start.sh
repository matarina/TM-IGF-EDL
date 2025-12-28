python train.py --pdb_dir data/pdb_data --ms_csv data/ms_intensity.csv --esm_dir esm2_t33_650M_UR50D --batch_size 4 --max_epochs 40 --fusion_iters 2 --hidden_dim 512 --lr 3e-4 --lr_esm 5e-5 --weight_decay 1e-2 --kl_weight 1e-3 --kl_warmup_epochs 10 --num_workers 4



