### MPT

Code for paper

#### Acknowledgements
This code is inspired by Jiezhong Qiu et al.'s GCC: 
https://github.com/THUDM/GCC

Environment
```
Linux with Python=3.5
PyTorch=1.4.0
DGL=0.4.3post2
Networkx=2.3
```

how to run

Download Pretrain dataset
https://drive.google.com/file/d/1-ZZrPtjIYGuNcXRFR7qfzjb7eZ1IDOp4/view?usp=sharing


Pre-train part:
```
for predataset in 'livejournal' 'euall';
do
python train_motif_regression.py --exp Pretrain --model-path $predataset --tb-path tensorboard --gpu 1 --batch-size 256
done
```

Downstream Tasks:

graph classification:

Freeze Mode 
```
for dataset in 'imdb-binary' 'imdb-multi' 'rdt-b' 'rdt-5k' 'collab';
do
    bash scripts/generate.sh 1 $predataset/Pretrain_moco_False_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_256_hid_64_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999/current.pth $dataset
    bash scripts/graph_classification/ours.sh $predataset/Pretrain_moco_False_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_256_hid_64_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999 64 $dataset
done
```
Fine-Tune Mode
```
bash scripts/finetune.sh euall/Pretrain_moco_False_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_256_hid_64_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999/ 1 imdb-binary
````


data link:
https://drive.google.com/drive/folders/1-YwFiB_UaKFBMLbmbmTmlCD2oqEoiGCa?usp=sharing
