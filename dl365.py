from utils.general import Path, download

# Make Directories
dir = Path('../datasets/Objects365')  # dataset root dir
for p in 'images', 'labels':
    (dir / p).mkdir(parents=True, exist_ok=True)
    for q in 'train', 'val':
        (dir / p / q).mkdir(parents=True, exist_ok=True)

# Train, Val Splits
for split, patches in [('train', 50 + 1), ('val', 43 + 1)]:
    print(f"Processing {split} in {patches} patches ...")
    images, labels = dir / 'images' / split, dir / 'labels' / split

    # Download
    url = f"https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/{split}/"
    if split == 'train':
        download([f'{url}zhiyuan_objv2_{split}.tar.gz'], dir=dir, delete=False)  # annotations json
        download([f'{url}patch{i}.tar.gz' for i in range(patches)], dir=images, curl=True, delete=False, threads=8)
    elif split == 'val':
        download([f'{url}zhiyuan_objv2_{split}.json'], dir=dir, delete=False)  # annotations json
        download([f'{url}images/v1/patch{i}.tar.gz' for i in range(15 + 1)], dir=images, curl=True, delete=False, threads=8)
        download([f'{url}images/v2/patch{i}.tar.gz' for i in range(16, patches)], dir=images, curl=True, delete=False, threads=8)
