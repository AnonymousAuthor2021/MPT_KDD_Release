for predataset in 'livejournal' 'facebook' 'dblp_netrep' 'dblp_snap' 'academia';
do
for dataset in 'loan' 'investor' 'agent' 'dx';
do
    bash scripts/generate.sh 0 $predataset/Pretrain_moco_False_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_256_hid_64_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999/current.pth $dataset
    bash scripts/node_classification/ours.sh $predataset/Pretrain_moco_False_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_256_hid_64_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999 64 $dataset
done
done