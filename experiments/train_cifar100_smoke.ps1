python main.py `
    --model shvit_s1 `
    --data-set CIFAR `
    --data-path datasets/cifar100 `
    --input-size 224 `
    --epochs 1 `
    --batch-size 32 `
    --num_workers 2 `
    --output_dir results/cifar100_smoke `
    --no-model-ema