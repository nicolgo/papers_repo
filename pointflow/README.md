The repos cloned as submodule from [pointflow](https://github.com/stevenygd/PointFlow.git)
```
git submodule add https://github.com/stevenygd/PointFlow.git
```
## reproduce steps:
1. add dataset
2. run train.py
```
python train.py --log_name=gen/shapenet15k-cate_airplane --lr=2e-3 --dataset_type=shapenet15k --data_dir=data/ShapeNetCore.v2.PC15k --cates=airplane --dims=512-512-512 --latent_dims=256-256 --num_blocks=1 --latent_num_blocks=1 --batch_size=16 --zdim=128 --epochs=4000 --save_freq=50 --viz_freq=1 --log_freq=1 --val_freq=10 --use_latent_flow
```
3. run `demo.py`
```

```