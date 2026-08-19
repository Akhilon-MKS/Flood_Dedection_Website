from dataset import Sen1Floods11Dataset

dataset = Sen1Floods11Dataset(r"archive (1)\v1.2", img_size=64)
print(f"samples={len(dataset)}")

sar, label = dataset[0]
print(f"sar_shape={tuple(sar.shape)}")
print(f"label_shape={tuple(label.shape)}")
print(f"sar_range=({sar.min().item():.6f}, {sar.max().item():.6f})")
print(f"label_range=({label.min().item():.6f}, {label.max().item():.6f})")

assert len(dataset) == 446
assert tuple(sar.shape) == (2, 64, 64)
assert tuple(label.shape) == (1, 64, 64)
assert 0.0 <= sar.min() <= sar.max() <= 1.0
assert 0.0 <= label.min() <= label.max() <= 1.0