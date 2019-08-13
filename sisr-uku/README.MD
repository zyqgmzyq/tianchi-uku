# Generate npy data
```
python gen_data.py
```

# Training

```
cd /sisr-uku/
CUDA_VISIBLE_DEVICES=0 python train.py --ckpt False
```


# Testing
```
cd /sisr-uku/
python infer.py --ensemble --output_path /smart/submit/testb
```


