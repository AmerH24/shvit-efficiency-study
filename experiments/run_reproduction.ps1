$COMMON_ARGS = @(
    "--data-set", "CIFAR",
    "--data-path", "datasets/cifar100",
    "--input-size", "128",
    "--epochs", "15",
    "--batch-size", "64",
    "--num_workers", "2",
    "--no-repeated-aug",
    "--no-model-ema",
    "--warmup-epochs", "1",
    "--cooldown-epochs", "0",
    "--mixup", "0",
    "--cutmix", "0",
    "--save_freq", "15"
)

Write-Host "=========================================="
Write-Host "1/4 - SHViT partial ratio 1/8"
Write-Host "=========================================="

python main.py `
    --model shvit_s1_ratio_1_8 `
    --output_dir results/ratio_1_8 `
    @COMMON_ARGS


Write-Host "=========================================="
Write-Host "2/4 - SHViT default ratio ~1/4.67"
Write-Host "=========================================="

python main.py `
    --model shvit_s1_ratio_default `
    --output_dir results/ratio_default `
    @COMMON_ARGS


Write-Host "=========================================="
Write-Host "3/4 - SHViT partial ratio 1/2"
Write-Host "=========================================="

python main.py `
    --model shvit_s1_ratio_1_2 `
    --output_dir results/ratio_1_2 `
    @COMMON_ARGS


Write-Host "=========================================="
Write-Host "4/4 - Progressive stage-wise extension"
Write-Host "=========================================="

python main.py `
    --model shvit_s1_progressive `
    --output_dir results/progressive `
    @COMMON_ARGS


Write-Host "=========================================="
Write-Host "All experiments completed."
Write-Host "=========================================="