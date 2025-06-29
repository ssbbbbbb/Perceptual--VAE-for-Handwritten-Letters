# Perceptual--VAE-for-Handwritten-Letters

# 1. 下載 EMNIST letters 資料集

```
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
```

使用 tensorflow_datasets 下載 EMNIST 字母手寫圖像資料集，分為訓練集和測試集。

