"""
tf.data pipeline cho MSI-Net.
Khớp với data.py gốc:
  - Resize giữ aspect ratio (area khi shrink, bicubic khi enlarge)
  - Symmetric padding (126 cho ảnh RGB, 0 cho saliency)
  - Tạo 3 scale inputs: full, 1/2, 1/4
"""
import os, tensorflow as tf

def _get_file_list(path):
    files = []
    for root, _, fs in os.walk(path):
        for f in sorted(fs):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(root, f))
    if not files:
        raise FileNotFoundError(f'Không có ảnh tại: {path}')
    return sorted(files)

def _resize_keep_aspect(image, target_h, target_w):
    """Resize giữ aspect ratio — dùng area khi shrink, bicubic khi enlarge."""
    cur = tf.shape(image)[:2]
    h_ratio = tf.cast(target_h, tf.float64) / tf.cast(cur[0], tf.float64)
    w_ratio = tf.cast(target_w, tf.float64) / tf.cast(cur[1], tf.float64)
    ratio   = tf.minimum(h_ratio, w_ratio)
    new_h   = tf.cast(tf.round(tf.cast(cur[0], tf.float64) * ratio), tf.int32)
    new_w   = tf.cast(tf.round(tf.cast(cur[1], tf.float64) * ratio), tf.int32)
    new_size = tf.stack([new_h, new_w])
    shrinking = tf.logical_or(cur[0] > new_h, cur[1] > new_w)
    img4d = tf.expand_dims(image, 0)
    resized = tf.cond(
        shrinking,
        lambda: tf.image.resize(img4d, new_size, method='area'),
        lambda: tf.image.resize(img4d, new_size, method='bicubic')
    )
    return tf.clip_by_value(resized[0], 0.0, 255.0)

def _pad_to_target(image, target_h, target_w):
    """Padding đối xứng — 126 cho RGB, 0 cho saliency."""
    cur = tf.shape(image)
    is_rgb = tf.equal(cur[2], 3)
    pad_val = tf.cond(is_rgb, lambda: 126.0, lambda: 0.0)
    pad_v = tf.cast(target_h - cur[0], tf.float32) / 2.0
    pad_h = tf.cast(target_w - cur[1], tf.float32) / 2.0
    padding = [
        [tf.cast(tf.math.floor(pad_v), tf.int32), tf.cast(tf.math.ceil(pad_v), tf.int32)],
        [tf.cast(tf.math.floor(pad_h), tf.int32), tf.cast(tf.math.ceil(pad_h), tf.int32)],
        [0, 0],
    ]
    return tf.pad(image, padding, constant_values=pad_val)

def _parse(img_path, sal_path, h, w):
    """Load, resize (aspect-ratio), pad, normalize — rồi tạo 3 scale."""
    # ── Image ──────────────────────────────────────────────────────────
    img_raw = tf.io.read_file(img_path)
    img = tf.cond(tf.image.is_jpeg(img_raw),
                  lambda: tf.image.decode_jpeg(img_raw, channels=3),
                  lambda: tf.image.decode_png(img_raw,  channels=3))
    img = tf.cast(img, tf.float32)
    img = _resize_keep_aspect(img, h, w)
    img = _pad_to_target(img, h, w)
    img = img / 255.0
    img = tf.ensure_shape(img, [h, w, 3])

    # ── Saliency map ────────────────────────────────────────────────────
    sal_raw = tf.io.read_file(sal_path)
    sal = tf.cond(tf.image.is_jpeg(sal_raw),
                  lambda: tf.image.decode_jpeg(sal_raw, channels=1),
                  lambda: tf.image.decode_png(sal_raw,  channels=1))
    sal = tf.cast(sal, tf.float32)
    sal = _resize_keep_aspect(sal, h, w)
    sal = _pad_to_target(sal, h, w)
    sal = sal / 255.0
    sal = tf.ensure_shape(sal, [h, w, 1])

    # ── Tạo 3 scale inputs cho MSI ──────────────────────────────────────
    img_half    = tf.image.resize(img, [h//2, w//2], method='area')
    img_quarter = tf.image.resize(img, [h//4, w//4], method='area')

    return (img, img_half, img_quarter), sal

def build_dataset(img_dir, sal_dir, target_size, batch_size,
                  shuffle=True, buffer_size=1000):
    imgs = _get_file_list(img_dir)
    sals = _get_file_list(sal_dir)
    assert len(imgs) == len(sals), f'Mismatch: {len(imgs)} imgs vs {len(sals)} sals'
    h, w = target_size
    ds = tf.data.Dataset.from_tensor_slices((imgs, sals))
    if shuffle:
        ds = ds.shuffle(buffer_size, seed=42)
    ds = ds.map(lambda x, y: _parse(x, y, h, w),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(imgs)
