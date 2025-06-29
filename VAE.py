import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

# ========================
# 1. 下載 EMNIST letters
# ========================
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# ========================
# 2. 預處理: 正規化
# ========================

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # (28,28,1)
    return image, image

ds_train = ds_train.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
ds_test  = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

# ========================
# Perceptual Model
# ========================
vgg = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet', input_shape=(32,32,3)
)
vgg.trainable = False
perceptual_layer = tf.keras.Model(
    inputs=vgg.input,
    outputs=vgg.get_layer('block3_conv3').output
)

# ========================
# 3. Conv Encoder + U-Net-style Decoder
# ========================
latent_dim = 4

class Encoder(layers.Layer):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(256, activation='relu')
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

    def call(self, x):
        skips = []
        x = self.conv1(x); skips.append(x)
        x = self.conv2(x); skips.append(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        return z, z_mean, z_log_var, skips

class Decoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.fc = layers.Dense(4*4*128, activation='relu')
        self.reshape_layer = layers.Reshape((4,4,128))
        self.up1 = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')
        self.up2 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.up3 = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')
        self.crop = layers.Cropping2D(((2,2),(2,2)))
        self.out_conv = layers.Conv2D(1, 3, padding='same', activation='sigmoid')

    def call(self, z, skips):
        x = self.fc(z)
        x = self.reshape_layer(x)
        x = self.up1(x)
        skip1 = tf.pad(skips[-1], [[0,0],[0,1],[0,1],[0,0]])
        x = tf.concat([x, skip1], axis=-1)
        x = self.up2(x)
        skip2 = tf.pad(skips[-2], [[0,0],[1,1],[1,1],[0,0]])
        x = tf.concat([x, skip2], axis=-1)
        x = self.up3(x)
        x = self.crop(x)
        return self.out_conv(x)

# ========================
# 4. β-VAE with Perceptual Loss
# ========================
class VAE(Model):
    def __init__(self, encoder, decoder, beta=500.0, perc_weight=0.1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.constant(beta, dtype=tf.float32)
        self.perc_weight = tf.constant(perc_weight, dtype=tf.float32)
        self.total_tracker = tf.keras.metrics.Mean(name='total')
        self.recon_tracker = tf.keras.metrics.Mean(name='recon')
        self.kl_tracker = tf.keras.metrics.Mean(name='kl')
        self.perc_tracker = tf.keras.metrics.Mean(name='perc_loss')
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')

    def compute_loss(self, x):
        z, zm, zv, skips = self.encoder(x)
        xr = self.decoder(z, skips)
        # Reconstruction loss
        per_px = self.bce(x, xr)
        per_px = tf.reshape(per_px, (tf.shape(per_px)[0], -1))
        recon = tf.reduce_mean(tf.reduce_sum(per_px, axis=1))
        # KL loss
        kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + zv - tf.square(zm) - tf.exp(zv), axis=1))
        # Perceptual loss
        x_rgb = tf.image.grayscale_to_rgb(x)
        xr_rgb = tf.image.grayscale_to_rgb(xr)
        x_resized = tf.image.resize(x_rgb, (32,32))
        xr_resized = tf.image.resize(xr_rgb, (32,32))
        feat_x = perceptual_layer(x_resized)
        feat_xr = perceptual_layer(xr_resized)
        perc_loss = tf.reduce_mean(tf.square(feat_x - feat_xr))
        # total
        loss = recon + self.beta * kl + self.perc_weight * perc_loss
        return loss, recon, kl, perc_loss

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            loss, recon, kl, perc_loss = self.compute_loss(x)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # update metrics
        self.total_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        self.perc_tracker.update_state(perc_loss)
        return {'total': self.total_tracker.result(), 'recon': self.recon_tracker.result(), 'kl': self.kl_tracker.result(), 'perc_loss': self.perc_tracker.result()}

    def test_step(self, data):
        x, _ = data
        loss, recon, kl, perc_loss = self.compute_loss(x)
        self.total_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        self.perc_tracker.update_state(perc_loss)
        return {'total': self.total_tracker.result(), 'recon': self.recon_tracker.result(), 'kl': self.kl_tracker.result(), 'perc_loss': self.perc_tracker.result()}

# ========================
# 5. Training
# ========================
encoder = Encoder(latent_dim)
decoder = Decoder()
vae = VAE(encoder, decoder, beta=500.0, perc_weight=0.1)
vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

# 訓練模型，記錄 history
history = vae.fit(
    ds_train,
    epochs=30,
    validation_data=ds_test
)

# ========================
# 6. 畫出loss曲線
# ========================
loss_keys = ['total', 'recon', 'kl', 'perc_loss']
plt.figure(figsize=(15, 8))
for i, key in enumerate(loss_keys):
    plt.subplot(2, 2, i + 1)
    plt.plot(history.history[key], label=f"train_{key}")
    plt.plot(history.history['val_' + key], label=f"val_{key}")
    plt.title(f"{key} loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
plt.tight_layout()
plt.show()

# ========================
# 7. 可視化 (重建效果)
# ========================
for test_batch, _ in ds_test.take(1):
    z, _, _, s = vae.encoder(test_batch)
    recon = vae.decoder(z, s)
    plt.figure(figsize=(20,4))
    for i in range(10):
        ax = plt.subplot(2,10,i+1)
        plt.imshow(test_batch[i].numpy().squeeze(), cmap='gray'); plt.axis('off')
        ax = plt.subplot(2,10,i+11)
        plt.imshow(recon[i].numpy().squeeze(), cmap='gray'); plt.axis('off')
    plt.show()
