Train the model on AMASS:
```bash
PYTHONPATH=. python exps/amass/train_progressive_model.py \
--data_dir /media/kaseris/FastData3/amass/ \
--model_cfg configs/amass/baseline.yml \
--gru_ae_ckpt pretrained_models/amass/gru_velocity_amass.pt 
```

Train the model on H3.6m:
```bash
PYTHONPATH=. python exps/h36m/train_progressive_model.py \
--data_dir datasets \
--model_cfg configs/h36m/baseline.yml \
--gru_ae_ckpt pretrained_models/h36m/gru_velocity.pt 
```

Test the model on H3.6m:

```bash
PYTHONPATH=. python exps/h36m/test_progressive_model.py \
--main_model_chkpt pretrained_models/h36m/checkpoint_iter_50000_20241002_172848.pt \
--ae_ckpt pretrained_models/h36m/gru_velocity.pt \
--data_dir datasets
--model_cfg configs/h36m/baseline.yml
```