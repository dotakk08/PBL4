"""
MSI-Net: Multi-Scale Input Network for Visual Saliency Prediction.
Khớp với kiến trúc gốc:
  - Encoder VGG16-style (13 conv layers)
  - Multi-Scale Input: ảnh full + 1/2 + 1/4 được encode riêng, features concat
  - Decoder: UpSampling 8x + Conv1x1 sigmoid
  - Load pretrained VGG16-Hybrid weights (nếu có)
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os

# ── Tên các layer VGG16 theo thứ tự để map pretrained weights ────────────
VGG_LAYER_NAMES = [
    'conv1_1','conv1_2',
    'conv2_1','conv2_2',
    'conv3_1','conv3_2','conv3_3',
    'conv4_1','conv4_2','conv4_3',
    'conv5_1','conv5_2','conv5_3',
]

def _vgg_block(x, filters, n_convs, block_idx):
    """Một VGG block gồm n_convs conv layers + MaxPool."""
    for i in range(1, n_convs + 1):
        name = f'conv{block_idx}_{i}'
        x = layers.Conv2D(filters, 3, activation='relu', padding='same', name=name)(x)
    x = layers.MaxPooling2D(2, 2, padding='same', name=f'pool{block_idx}')(x)
    return x

def _vgg_encoder(inp, suffix=''):
    """
    VGG16-style encoder (conv1→conv5, không có pool5).
    suffix: dùng để phân biệt tên layer giữa 3 scale ('', '_half', '_quarter')
    """
    def conv(x, f, name):
        return layers.Conv2D(f, 3, activation='relu', padding='same',
                             name=name+suffix)(x)
    def pool(x, name):
        return layers.MaxPooling2D(2, 2, padding='same', name=name+suffix)(x)

    x = conv(inp, 64,  'conv1_1'); x = conv(x, 64,  'conv1_2'); x = pool(x, 'pool1')
    x = conv(x,  128, 'conv2_1'); x = conv(x, 128, 'conv2_2'); x = pool(x, 'pool2')
    x = conv(x,  256, 'conv3_1'); x = conv(x, 256, 'conv3_2')
    x = conv(x,  256, 'conv3_3'); x = pool(x, 'pool3')
    x = conv(x,  512, 'conv4_1'); x = conv(x, 512, 'conv4_2')
    x = conv(x,  512, 'conv4_3'); x = pool(x, 'pool4')
    x = conv(x,  512, 'conv5_1'); x = conv(x, 512, 'conv5_2')
    x = conv(x,  512, 'conv5_3')
    # Không có pool5 — giữ spatial resolution để upsample sau
    return x

def build_msinet(input_shape=(240, 320, 3)):
    """
    Xây dựng MSI-Net với 3 nhánh encoder (full, 1/2, 1/4 scale).
    Feature maps từ 3 nhánh được upsample về cùng kích thước rồi concat.
    Decoder: UpSampling 8× + Conv1x1 sigmoid + Resizing về input size.
    """
    h, w = input_shape[0], input_shape[1]

    # ── 3 input tensors ở 3 scale ──────────────────────────────────────
    inp_full    = tf.keras.Input(shape=input_shape,          name='input_full')
    inp_half    = tf.keras.Input(shape=(h//2, w//2, 3),      name='input_half')
    inp_quarter = tf.keras.Input(shape=(h//4, w//4, 3),      name='input_quarter')

    # ── 3 nhánh encoder ────────────────────────────────────────────────
    feat_full    = _vgg_encoder(inp_full,    suffix='')
    feat_half    = _vgg_encoder(inp_half,    suffix='_half')
    feat_quarter = _vgg_encoder(inp_quarter, suffix='_quarter')

    # ── Upsample half & quarter về cùng spatial size với full ──────────
    # Dùng Lambda lấy shape động từ feat_full để tránh lỗi khi h/w không
    # chia hết cho 16 (vd: 360/16=22.5 → full=23, half=22 → mismatch)
    feat_half_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_half')([feat_half, feat_full])
    feat_quarter_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_quarter')([feat_quarter, feat_full])

    # ── Concat 3 nhánh → 512×3 = 1536 channels ─────────────────────────
    merged = layers.Concatenate(name='concat_scales')(
                [feat_full, feat_half_up, feat_quarter_up])

    # ── Decoder ─────────────────────────────────────────────────────────
    x = layers.UpSampling2D(size=(8, 8), name='upsample')(merged)
    x = layers.Conv2D(1, 1, activation='sigmoid',
                      padding='same', name='output')(x)
    out = layers.Resizing(h, w, name='resize_output')(x)

    model = Model(inputs=[inp_full, inp_half, inp_quarter],
                  outputs=out, name='MSINet')
    return model


def load_vgg16_hybrid_weights(model, weights_path):
    """
    Load pretrained VGG16-Hybrid weights vào các nhánh encoder.
    weights_path: đường dẫn đến file .npy hoặc .h5 chứa VGG16 weights.
    Weights được copy sang cả 3 nhánh (full, half, quarter).
    """
    if not os.path.exists(weights_path):
        print(f'  ⚠️  Không tìm thấy weights: {weights_path} — bỏ qua load pretrained')
        return

    print(f'  Đang load VGG16-Hybrid weights từ {weights_path} ...')
    try:
        w = np.load(weights_path, allow_pickle=True).item()
    except Exception:
        try:
            import h5py
            w = {}
            with h5py.File(weights_path, 'r') as f:
                for k in f.keys():
                    w[k] = [np.array(f[k][wk]) for wk in f[k].keys()]
        except Exception as e:
            print(f'  ⚠️  Không đọc được weights: {e}')
            return

    suffixes = ['', '_half', '_quarter']
    copied = 0
    for name in VGG_LAYER_NAMES:
        if name not in w:
            continue
        for sfx in suffixes:
            layer_name = name + sfx
            try:
                layer = model.get_layer(layer_name)
                layer.set_weights(w[name])
                copied += 1
            except (ValueError, KeyError):
                pass
    print(f'  ✅ Copied pretrained weights vào {copied} layers')
