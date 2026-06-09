import tensorflow as tf
from tensorflow.keras import layers, Model

_EFNET_OUTPUT_LAYER = 'block5a_project_bn'   # stride=16, channels=112


def _efficientnet_encoder(inp, name_prefix):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inp,
        pooling=None,
    )
    feat = base.get_layer(_EFNET_OUTPUT_LAYER).output
    encoder = Model(inputs=base.input, outputs=feat, name=f'encoder_{name_prefix}')
    return encoder(inp)


def build_msinet(input_shape=(240, 320, 3)):
    h, w = input_shape[0], input_shape[1]

    inp_full    = tf.keras.Input(shape=input_shape,     name='input_full')
    inp_half    = tf.keras.Input(shape=(h//2, w//2, 3), name='input_half')
    inp_quarter = tf.keras.Input(shape=(h//4, w//4, 3), name='input_quarter')

    feat_full    = _efficientnet_encoder(inp_full,    name_prefix='full')
    feat_half    = _efficientnet_encoder(inp_half,    name_prefix='half')
    feat_quarter = _efficientnet_encoder(inp_quarter, name_prefix='quarter')

    feat_half_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_half')([feat_half, feat_full])
    feat_quarter_up = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name='resize_quarter')([feat_quarter, feat_full])

    # 112×3 = 336 channels
    merged = layers.Concatenate(name='concat_scales')(
                [feat_full, feat_half_up, feat_quarter_up])

    # stride=16 → UpSampling 16× (giống VGG16 baseline)
    x = layers.UpSampling2D(size=(16, 16), name='upsample')(merged)
    x = layers.Conv2D(1, 1, activation='sigmoid', padding='same', name='output')(x)
    out = layers.Resizing(h, w, name='resize_output')(x)

    model = Model(inputs=[inp_full, inp_half, inp_quarter],
                  outputs=out, name='MSINet_EfficientB0')
    return model


def load_efficientnet_weights(model):
    print('  ✅ EfficientNetB0 backbone dùng ImageNet weights (load tự động qua Keras).')
