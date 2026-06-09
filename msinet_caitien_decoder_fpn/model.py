"""
MSI-Net V2: VGG16-Hybrid Encoder + FPN-style Decoder với skip connections.

Thay đổi so với baseline:
- Encoder: giữ nguyên VGG16-style (conv1→conv5), load pretrained VGG16-Hybrid weights
- MSI: giữ nguyên 3 nhánh encoder song song (full, 1/2, 1/4)
- Decoder: NÂNG CẤP từ UpSampling 8× → FPN-style với skip connections
    * concat features từ 3 MSI branches (1536ch) → bottleneck Conv 512
    * skip từ pool3 level (avg của 3 nhánh, 256ch) → fuse sau up2x
    * skip từ pool2 level (avg của 3 nhánh, 128ch) → fuse sau up4x
    * skip từ pool1 level (avg của 3 nhánh, 64ch)  → fuse sau up8x
    * Conv 1×1 sigmoid → resize to input
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np, os

VGG_LAYER_NAMES = [
    'conv1_1','conv1_2',
    'conv2_1','conv2_2',
    'conv3_1','conv3_2','conv3_3',
    'conv4_1','conv4_2','conv4_3',
    'conv5_1','conv5_2','conv5_3',
]

def _vgg_encoder(inp, suffix=''):
    """VGG16-style encoder — giữ nguyên như baseline."""
    def conv(x, f, name):
        return layers.Conv2D(f, 3, activation='relu', padding='same',
                             name=name+suffix)(x)
    def pool(x, name):
        return layers.MaxPooling2D(2, 2, padding='same', name=name+suffix)(x)

    x = conv(inp, 64,  'conv1_1'); x = conv(x, 64,  'conv1_2')
    p1 = pool(x, 'pool1')   # /2  — 64ch  → skip level 1
    x = conv(p1, 128, 'conv2_1'); x = conv(x, 128, 'conv2_2')
    p2 = pool(x, 'pool2')   # /4  — 128ch → skip level 2
    x = conv(p2, 256, 'conv3_1'); x = conv(x, 256, 'conv3_2')
    x = conv(x,  256, 'conv3_3')
    p3 = pool(x, 'pool3')   # /8  — 256ch → skip level 3
    x = conv(p3, 512, 'conv4_1'); x = conv(x, 512, 'conv4_2')
    x = conv(x,  512, 'conv4_3')
    p4 = pool(x, 'pool4')   # /16 — 512ch
    x = conv(p4, 512, 'conv5_1'); x = conv(x, 512, 'conv5_2')
    x = conv(x,  512, 'conv5_3')  # /16 — 512ch (no pool5)
    return x, p3, p2, p1   # feat_out, skip3, skip2, skip1

def _fpn_block(x, skip, filters, name, target_h, target_w):
    """FPN decode block: upsample x 2×, fuse with skip, refine."""
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear',
                             name=f'up_{name}')(x)
    # Resize skip về đúng shape của x (tránh off-by-one)
    x = layers.Resizing(target_h, target_w, name=f'resize_x_{name}')(x)
    skip_r = layers.Resizing(target_h, target_w, name=f'resize_skip_{name}')(skip)
    skip_r = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                           name=f'proj_skip_{name}')(skip_r)
    skip_r = layers.Activation('relu', name=f'act_skip_{name}')(skip_r)
    x = layers.Add(name=f'add_{name}')([x[:,:,:,:filters] if x.shape[-1]==filters
                                         else layers.Conv2D(filters, 1, padding='same',
                                                use_bias=False, name=f'proj_x_{name}')(x),
                                         skip_r])
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      use_bias=False, name=f'refine_{name}')(x)
    return x

def build_msinet(input_shape=(240, 320, 3)):
    """
    MSI-Net V2 với FPN decoder.
    Giữ nguyên: encoder VGG16-style + 3-branch MSI.
    Nâng cấp: decoder FPN với skip connections từ pool1/2/3.
    """
    h, w = input_shape[0], input_shape[1]

    # ── 3 inputs ────────────────────────────────────────────────────
    inp_full    = tf.keras.Input(shape=input_shape,          name='input_full')
    inp_half    = tf.keras.Input(shape=(h//2, w//2, 3),      name='input_half')
    inp_quarter = tf.keras.Input(shape=(h//4, w//4, 3),      name='input_quarter')

    # ── 3 nhánh VGG encoder — mỗi nhánh trả về (feat, skip3, skip2, skip1) ─
    feat_f, s3_f, s2_f, s1_f = _vgg_encoder(inp_full,    suffix='')
    feat_h, s3_h, s2_h, s1_h = _vgg_encoder(inp_half,    suffix='_half')
    feat_q, s3_q, s2_q, s1_q = _vgg_encoder(inp_quarter, suffix='_quarter')

    # ── Align half & quarter features về cùng spatial với full ─────
    # Shape full feat: (h//16, w//16) — dùng resize dynamic
    feat_h_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_feat_half')([feat_h, feat_f])
    feat_q_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_feat_quarter')([feat_q, feat_f])

    # ── Concat 3 branches → 1536ch → bottleneck 512 ────────────────
    merged = layers.Concatenate(name='concat_scales')(
                [feat_f, feat_h_up, feat_q_up])
    x = layers.Conv2D(512, 1, padding='same', activation='relu',
                      name='bottleneck')(merged)

    # ── Skip connections: average pool-level features từ 3 nhánh ───
    # pool3-level: /8 → 256ch
    s3_h_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_s3_half')([s3_h, s3_f])
    s3_q_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_s3_quarter')([s3_q, s3_f])
    skip3 = layers.Average(name='avg_skip3')([s3_f, s3_h_up, s3_q_up])

    # pool2-level: /4 → 128ch
    s2_h_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_s2_half')([s2_h, s2_f])
    s2_q_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_s2_quarter')([s2_q, s2_f])
    skip2 = layers.Average(name='avg_skip2')([s2_f, s2_h_up, s2_q_up])

    # pool1-level: /2 → 64ch
    s1_h_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_s1_half')([s1_h, s1_f])
    s1_q_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_s1_quarter')([s1_q, s1_f])
    skip1 = layers.Average(name='avg_skip1')([s1_f, s1_h_up, s1_q_up])

    # ── FPN Decoder ──────────────────────────────────────────────────
    # x: (h//16, w//16, 512) → up 2× → (h//8, w//8)
    x = layers.UpSampling2D(size=(2,2), interpolation='bilinear', name='up_d4')(x)
    x = layers.Resizing(h//8, w//8, name='resize_d4')(x)
    skip3_p = layers.Conv2D(256, 1, padding='same', activation='relu',
                             use_bias=False, name='proj_skip3')(skip3)
    skip3_r = layers.Resizing(h//8, w//8, name='resize_skip3')(skip3_p)
    x_proj  = layers.Conv2D(256, 1, padding='same', use_bias=False, name='proj_x_d4')(x)
    x = layers.Add(name='add_d4')([x_proj, skip3_r])
    x = layers.Conv2D(256, 3, padding='same', activation='relu',
                      use_bias=False, name='refine_d4')(x)

    # x: (h//8, w//8, 256) → up 2× → (h//4, w//4)
    x = layers.UpSampling2D(size=(2,2), interpolation='bilinear', name='up_d3')(x)
    x = layers.Resizing(h//4, w//4, name='resize_d3')(x)
    skip2_p = layers.Conv2D(128, 1, padding='same', activation='relu',
                             use_bias=False, name='proj_skip2')(skip2)
    skip2_r = layers.Resizing(h//4, w//4, name='resize_skip2')(skip2_p)
    x_proj  = layers.Conv2D(128, 1, padding='same', use_bias=False, name='proj_x_d3')(x)
    x = layers.Add(name='add_d3')([x_proj, skip2_r])
    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                      use_bias=False, name='refine_d3')(x)

    # x: (h//4, w//4, 128) → up 2× → (h//2, w//2)
    x = layers.UpSampling2D(size=(2,2), interpolation='bilinear', name='up_d2')(x)
    x = layers.Resizing(h//2, w//2, name='resize_d2')(x)
    skip1_p = layers.Conv2D(64, 1, padding='same', activation='relu',
                             use_bias=False, name='proj_skip1')(skip1)
    skip1_r = layers.Resizing(h//2, w//2, name='resize_skip1')(skip1_p)
    x_proj  = layers.Conv2D(64, 1, padding='same', use_bias=False, name='proj_x_d2')(x)
    x = layers.Add(name='add_d2')([x_proj, skip1_r])
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      use_bias=False, name='refine_d2')(x)

    # x: (h//2, w//2, 64) → up 2× → full res
    x = layers.UpSampling2D(size=(2,2), interpolation='bilinear', name='up_d1')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu',
                      use_bias=False, name='refine_d1')(x)
    x = layers.Conv2D(1, 1, activation='sigmoid', padding='same', name='output')(x)
    out = layers.Resizing(h, w, name='resize_output')(x)

    model = Model(inputs=[inp_full, inp_half, inp_quarter],
                  outputs=out, name='MSINet_V2_FPN')
    return model


def load_vgg16_hybrid_weights(model, weights_path):
    """
    Load pretrained VGG16-Hybrid weights vào cả 3 nhánh encoder.
    Giống hệt baseline — chỉ encoder thay đổi, decoder học từ đầu.
    """
    if not os.path.exists(weights_path):
        print(f'  ⚠️  Không tìm thấy weights: {weights_path}')
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
            try:
                layer = model.get_layer(name + sfx)
                layer.set_weights(w[name])
                copied += 1
            except (ValueError, KeyError):
                pass
    print(f'  ✅ Copied pretrained weights vào {copied} layers')
