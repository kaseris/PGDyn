```bash
PYTHONPATH=. python exps/h36m/test_progressive_model.py \
--main_model_chkpt pretrained_models/h36m/checkpoint_iter_50000_20241002_172848.pt \
--ae_ckpt pretrained_models/h36m/gru_velocity.pt \
--data_dir datasets
```