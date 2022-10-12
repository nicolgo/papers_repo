import matplotlib.cm as cm
import matplotlib as mpl
from IPython.display import display_png
import math
import io
from clu import deterministic_data
import tensorflow_datasets as tfds
import jax
import jax.numpy as np
from jax import random, vmap, grad, jit
import numpy as onp
import matplotlib.pyplot as plt
import flax
import flax.linen as nn
import optax
import einops
from tqdm import tqdm, trange
import tensorflow_probability.substrates.jax as tfp
from scipy.stats import describe
import functools
from functools import partial
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')
tfd = tfp.distributions
tfb = tfp.bijectors


# @title utils
# Various helper utility functions.


def imify(arr, vmin=None, vmax=None, cmap=None, origin=None):
    """Convert an array to an image.

    Arguments:
      arr : array-like The image data. The shape can be one of MxN (luminance),
        MxNx3 (RGB) or MxNx4 (RGBA).
      vmin : scalar, optional lower value.
      vmax : scalar, optional *vmin* and *vmax* set the color scaling for the
        image by fixing the values that map to the colormap color limits. If
        either *vmin* or *vmax* is None, that limit is determined from the *arr*
        min/max value.
      cmap : str or `~matplotlib.colors.Colormap`, optional A Colormap instance or
        registered colormap name. The colormap maps scalar data to colors. It is
        ignored for RGB(A) data.
          Defaults to :rc:`image.cmap` ('viridis').
      origin : {'upper', 'lower'}, optional Indicates whether the ``(0, 0)`` index
        of the array is in the upper
          left or lower left corner of the axes.  Defaults to :rc:`image.origin`
            ('upper').

    Returns:
      A uint8 image array.
    """
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin, vmax)
    if origin is None:
        origin = mpl.rcParams["image.origin"]
    if origin == "lower":
        arr = arr[::-1]
    rgba = sm.to_rgba(arr, bytes=True)
    return rgba


def rawarrview(array, **kwargs):
    """Visualize an array as if it was an image in colab notebooks.

    Arguments:
      array: an array which will be turned into an image.
      **kwargs: Additional keyword arguments passed to imify.
    """
    f = io.BytesIO()
    imarray = imify(array, **kwargs)
    plt.imsave(f, imarray, format="png")
    f.seek(0)
    dat = f.read()
    f.close()
    display_png(dat, raw=True)


def reshape_image_batch(array, cut=None, rows=None, axis=0):
    """Given an array of shape [n, x, y, ...] reshape it to create an image field.

    Arguments:
      array: The array to reshape.
      cut: Optional cut on the number of images to view. Will default to whole
        array.
      rows: Number of rows to use.  Will default to the integer less than the
        sqrt.
      axis: Axis to interpretate at the batch dimension.  By default the image
        dimensions immediately follow.

    Returns:
      reshaped_array: An array of shape [rows * x, cut / rows * y, ...]
    """
    original_shape = array.shape
    assert len(original_shape) >= 2, "array must be at least 3 Dimensional."

    if cut is None:
        cut = original_shape[axis]
    if rows is None:
        rows = int(math.sqrt(cut))

    cols = cut // rows
    cut = cols * rows

    leading = original_shape[:axis]
    x_width = original_shape[axis + 1]
    y_width = original_shape[axis + 2]
    remaining = original_shape[axis + 3:]

    array = array[:cut]
    array = array.reshape(leading + (rows, cols, x_width, y_width) + remaining)
    array = np.moveaxis(array, axis + 2, axis + 1)
    array = array.reshape(
        leading + (rows * x_width, cols * y_width) + remaining)
    return array


def zoom(im, k, axes=(0, 1)):
    for ax in axes:
        im = np.repeat(im, k, ax)
    return im


def imgviewer(im, zoom=3, cmap='bone_r', normalize=False, **kwargs):
    if normalize:
        im = im - im.min()
        im = im / im.max()
    return rawarrview(zoom(im, zoom), cmap=cmap, **kwargs)


def param_count(pytree):
    return sum(x.size for x in jax.tree_leaves(pytree))


replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

# data


def preprocess_fn(example):
    image = tf.cast(example['image'], 'float32')
    image = tf.transpose(image, (1, 0, 2,))
    image = tf.random.uniform(image.shape) < image / 255.0
    return (image, example["label"] + 1)


def unconditional_fraction(example, p=0.2):
    image, label = example
    label = tf.where(tf.random.uniform(label.shape) > p, label, 0)
    return (image, label)


def create_input_iter(ds):
    def _prepare(xs):
        def _f(x):
            x = x._numpy()
            return x
        return jax.tree_util.tree_map(_f, xs)
    it = map(_prepare, ds)
    it = flax.jax_utils.prefetch_to_device(it, 2)
    return it


# forward process
def gamma(ts, gamma_min=-6, gamma_max=6):
    return gamma_max + (gamma_min - gamma_max)*ts


def sigma2(gamma):
    return jax.nn.sigmoid(-gamma)


def alpha(gamma):
    return np.sqrt(1-sigma2(gamma))


def variance_preserving_map(x, gamma, eps):
    a = alpha(gamma)
    var = sigma2(gamma)
    return a*x + np.sqrt(var)*eps


def show_forward_process(out):
    ts = np.linspace(0, 1, 1000)
    plt.plot(ts, sigma2(gamma(ts)), label=r'$\sigma^2$')
    plt.plot(ts, alpha(gamma(ts)), label=r'$\alpha$')
    plt.legend()

    ii = 2 * out[0][0, :10].squeeze() - 1
    ii.shape
    TT = 30
    results = np.zeros((10, TT+1, 28, 28))
    results = results.at[:, 0].set(ii)

    tt = np.linspace(0, 1, TT)
    eps = random.normal(jax.random.PRNGKey(0), (10, 28, 28))
    for i, t in enumerate(tt):
        results = results.at[:, i +
                             1].set(variance_preserving_map(ii, gamma(t), eps))
    rawarrview(reshape_image_batch(results.reshape(
        (-1, 28, 28)), rows=10), cmap='bone_r')


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=np.float32):
    """Build sinusoidal embeddings (from Fairseq)."""

    assert len(timesteps.shape) == 1
    timesteps *= 1000

    half_dim = embedding_dim // 2
    emb = np.log(10_000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class ResNet(nn.Module):
    hidden_size: int = 512
    n_layers: int = 1
    middle_size: int = 1024

    @nn.compact
    def __call__(self, x, cond=None):
        assert x.shape[-1] == self.hidden_size, "Input must be hidden size."
        z = x
        for i in range(self.n_layers):
            h = nn.gelu(nn.LayerNorm()(z))
            h = nn.Dense(self.middle_size)(h)
            if cond is not None:
                h += nn.Dense(self.middle_size, use_bias=False)(cond)
            h = nn.gelu(nn.LayerNorm()(h))
            h = nn.Dense(self.hidden_size,
                         kernel_init=jax.nn.initializers.zeros)(h)
            z = z + h
        return z


class Encoder(nn.Module):
    hidden_size: int = 256
    n_layers: int = 3
    z_dim: int = 128

    @nn.compact
    def __call__(self, ims, cond=None):
        x = 2 * ims.astype('float32') - 1.0
        x = einops.rearrange(x, '... x y d -> ... (x y d)')
        x = nn.Dense(self.hidden_size)(x)
        x = ResNet(self.hidden_size, self.n_layers)(x, cond=cond)
        params = nn.Dense(self.z_dim)(x)
        return params


class Decoder(nn.Module):
    hidden_size: int = 512
    n_layers: int = 3

    @nn.compact
    def __call__(self, z, cond=None):
        z = nn.Dense(self.hidden_size)(z)
        z = ResNet(self.hidden_size, self.n_layers)(z, cond=cond)
        logits = nn.Dense(28 * 28 * 1)(z)
        logits = einops.rearrange(
            logits, '... (x y d) -> ... x y d', x=28, y=28, d=1)
        return tfd.Independent(tfd.Bernoulli(logits=logits), 3)


class ScoreNet(nn.Module):
    embedding_dim: int = 128
    n_layers: int = 10

    @nn.compact
    def __call__(self, z, g_t, conditioning):
        n_embd = self.embedding_dim

        t = g_t
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(z.shape[0])  # ensure t is a vector

        temb = get_timestep_embedding(t, n_embd)
        cond = np.concatenate([temb, conditioning], axis=1)
        cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
        cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))
        cond = nn.Dense(n_embd)(cond)

        h = nn.Dense(n_embd)(z)
        h = ResNet(n_embd, self.n_layers)(h, cond)
        return z + h


class VDM(nn.Module):
    timesteps: int = 1000
    gamma_min: float = -3.0  # -13.3
    gamma_max: float = 3.0  # 5.0
    embedding_dim: int = 256
    antithetic_time_sampling: bool = True
    layers: int = 32
    classes: int = 10 + 26 + 26 + 1

    def setup(self):
        self.gamma = partial(gamma, gamma_min=self.gamma_min,
                             gamma_max=self.gamma_max)
        self.score_model = ScoreNet(n_layers=self.layers,
                                    embedding_dim=self.embedding_dim)
        self.encoder = Encoder(z_dim=self.embedding_dim)
        self.decoder = Decoder()
        self.embedding_vectors = nn.Embed(self.classes, self.embedding_dim)

    def gammat(self, t):
        return self.gamma(t)

    def recon_loss(self, x, f, cond):
        """The reconstruction loss measures the gap in the first step.

        We measure the gap from encoding the image to z_0 and back again."""
        # ## Reconsturction loss 2
        g_0 = self.gamma(0.0)
        eps_0 = random.normal(self.make_rng("sample"), shape=f.shape)
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)
        loss_recon = -self.decoder(z_0_rescaled, cond).log_prob(x.astype('int32'))
        return loss_recon

    def latent_loss(self, f):
        """The latent loss measures the gap in the last step, this is the KL
        divergence between the final sample from the forward process and starting 
        distribution for the reverse process, here taken to be a N(0,1)."""
        # KL z1 with N(0,1) prior
        g_1 = self.gamma(1.0)
        var_1 = sigma2(g_1)
        mean1_sqr = (1. - var_1) * np.square(f)
        loss_klz = 0.5 * np.sum(mean1_sqr + var_1 -
                                np.log(var_1) - 1., axis=-1)
        return loss_klz

    def diffusion_loss(self, t, f, cond):
        # sample z_t
        g_t = self.gamma(t)
        eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_t = variance_preserving_map(f, g_t[:, None], eps)
        # compute predicted noise
        eps_hat = self.score_model(z_t, g_t, cond)
        # compute MSE of predicted noise
        loss_diff_mse = np.sum(np.square(eps - eps_hat), axis=-1)

        # loss for finite depth T, i.e. discrete time
        T = self.timesteps
        s = t - (1./T)
        g_s = self.gamma(s)
        loss_diff = .5 * T * np.expm1(g_s - g_t) * loss_diff_mse
        return loss_diff

    def __call__(self, images, conditioning,
                 sample_shape=()):

        x = images
        n_batch = images.shape[0]

        cond = self.embedding_vectors(conditioning)

        # 1. RECONSTRUCTION LOSS
        # add noise and reconstruct
        f = self.encoder(x, cond)
        loss_recon = self.recon_loss(x, f, cond)

        # 2. LATENT LOSS
        # KL z1 with N(0,1) prior
        loss_klz = self.latent_loss(f)

        # 3. DIFFUSION LOSS
        # sample time steps
        rng1 = self.make_rng("sample")
        if self.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = np.mod(t0 + np.arange(0., 1., step=1. / n_batch), 1.0)
        else:
            t = jax.random.uniform(rng1, shape=(n_batch,))

        # discretize time steps if we're working with discrete time
        T = self.timesteps
        t = np.ceil(t * T) / T

        loss_diff = self.diffusion_loss(t, f, cond)

        # End of diffusion loss computation
        return (loss_diff, loss_klz, loss_recon)

    def embed(self, conditioning):
        return self.embedding_vectors(conditioning)

    def encode(self, ims, conditioning=None):
        cond = self.embedding_vectors(conditioning)
        return self.encoder(ims, cond)

    def decode(self, z0, conditioning=None):
        cond = self.embedding_vectors(conditioning)
        return self.decoder(z0, cond)

    def shortcut(self, ims, conditioning=None):
        "Evaluates the performance of the encoder / decoder by itself."
        cond = self.embedding_vectors(conditioning)
        f = self.encoder(ims, cond)
        eps_0 = random.normal(self.make_rng("sample"), shape=f.shape)
        g_0 = self.gamma(0.)
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)
        return self.decoder(z_0_rescaled, cond)

    def sample_step(self, rng, i, T, z_t, conditioning, guidance_weight=0.):
        rng_body = jax.random.fold_in(rng, i)
        eps = random.normal(rng_body, z_t.shape)
        t = (T - i)/T
        s = (T - i - 1) / T

        g_s = self.gamma(s)
        g_t = self.gamma(t)

        cond = self.embedding_vectors(conditioning)

        eps_hat_cond = self.score_model(
            z_t,
            g_t * np.ones((z_t.shape[0],), z_t.dtype),
            cond,)

        eps_hat_uncond = self.score_model(
            z_t,
            g_t * np.ones((z_t.shape[0],), z_t.dtype),
            cond * 0.,)
        eps_hat = (1. + guidance_weight) * eps_hat_cond - \
            guidance_weight * eps_hat_uncond

        a = nn.sigmoid(g_s)
        b = nn.sigmoid(g_t)
        c = -np.expm1(g_t - g_s)
        sigma_t = np.sqrt(sigma2(g_t))
        z_s = np.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) + \
            np.sqrt((1. - a) * c) * eps
        return z_s

    def recon(self, z, t, conditioning):
        g_t = self.gamma(t)[:, None]
        cond = self.embedding_vectors(conditioning)
        eps_hat = self.score_model(z, g_t, cond)
        sigmat = np.sqrt(sigma2(g_t))
        alphat = np.sqrt(1 - sigmat**2)
        xhat = (z - sigmat * eps_hat) / alphat
        return (eps_hat, xhat)


def generate(vdm, params, rng, shape, conditioning, guidance_weight=0.0):
    # first generate latent
    rng, spl = random.split(rng)
    zt = random.normal(spl, shape + (vdm.embedding_dim,))

    def body_fn(i, z_t):
        return vdm.apply(params, rng, i, vdm.timesteps, z_t,
                         conditioning, guidance_weight=guidance_weight, method=vdm.sample_step)

    z0 = jax.lax.fori_loop(
        lower=0, upper=vdm.timesteps, body_fun=body_fn, init_val=zt)
    g0 = vdm.apply(params, 0.0, method=vdm.gammat)
    var0 = sigma2(g0)
    z0_rescaled = z0 / np.sqrt(1. - var0)
    return vdm.apply(params, z0_rescaled, conditioning, method=vdm.decode)


def recon(vdm, params, rng, t, ims, conditioning):
    # first generate latent
    rng, spl = random.split(rng)
    z_0 = vdm.apply(params, ims, conditioning, method=vdm.encode)

    T = vdm.timesteps
    tn = np.ceil(t * T)
    t = tn / T
    g_t = vdm.apply(params, t, method=vdm.gammat)
    rng, spl = random.split(rng)
    eps = jax.random.normal(spl, shape=z_0.shape)
    z_t = variance_preserving_map(z_0, g_t[:, None], eps)

    def body_fn(i, z_t):
        return vdm.apply(
            params,
            rng,
            i,
            vdm.timesteps,
            z_t,
            conditioning,
            method=vdm.sample_step)

    z0 = jax.lax.fori_loop(
        lower=(T - tn).astype('int'),
        upper=vdm.timesteps, body_fun=body_fn, init_val=z_t)
    g0 = vdm.apply(params, 0.0, method=vdm.gammat)
    var0 = sigma2(g0)
    z0_rescaled = z0 / np.sqrt(1. - var0)
    return vdm.apply(params, z0_rescaled, conditioning, method=vdm.decode)


def elbo(vdm, params, rng, ims, conditioning):
    x = ims
    rng, spl = jax.random.split(rng)
    cond = vdm.apply(params, conditioning, method=vdm.embed)
    f = vdm.apply(params, ims, conditioning, method=vdm.encode)
    loss_recon = vdm.apply(params, x, f, cond, rngs={
                           "sample": rng}, method=vdm.recon_loss)
    loss_klz = vdm.apply(params, f, method=vdm.latent_loss)

    def body_fun(i, val):
        loss, rng = val
        rng, spl = jax.random.split(rng)
        new_loss = vdm.apply(params, np.array(
            [i / vdm.timesteps]), f, cond, rngs={"sample": spl}, method=vdm.diffusion_loss)
        return (loss + new_loss / vdm.timesteps, rng)

    loss_diff, rng = jax.lax.fori_loop(
        0, vdm.timesteps, body_fun, (np.zeros(ims.shape[0]), rng))

    return loss_recon + loss_klz + loss_diff


def viewer(x, **kwargs):
    # x = np.clip((x + 1)/2.0, 0, 1)
    return rawarrview(reshape_image_batch(x), **kwargs)


def main():
    print("devices:", jax.devices())
    dataset_builder = tfds.builder("emnist")
    dataset_builder.download_and_prepare()

    train_split = tfds.split_for_jax_process("train", drop_remainder=True)
    test_split = tfds.split_for_jax_process("test", drop_remainder=True)
    batch_size = 4*128*jax.device_count()

    train_ds = deterministic_data.create_dataset(dataset_builder, split=train_split,
                                                 rng=jax.random.PRNGKey(0), shuffle_buffer_size=100,
                                                 batch_dims=[jax.local_device_count(), batch_size //
                                                             jax.device_count()],
                                                 num_epochs=None, preprocess_fn=lambda x: unconditional_fraction(preprocess_fn(x)), shuffle=True)

    test_ds = deterministic_data.create_dataset(dataset_builder, split=test_split, rng=jax.random.PRNGKey(0),
                                                batch_dims=[jax.local_device_count(
                                                ), batch_size // jax.device_count()],
                                                num_epochs=1, preprocess_fn=preprocess_fn
                                                )
    out = next(create_input_iter(train_ds))
    # rawarrview(reshape_image_batch(out[0][0].squeeze()), cmap='bone_r')
    # forward process
    # show_forward_process(out)

    vdm = VDM(gamma_min=-5.0, gamma_max=1.0, layers=4, embedding_dim=32, timesteps=1000)
    rng = random.PRNGKey(0)

    batch = next(create_input_iter(train_ds))[0][0]
    conditioning = np.zeros(batch.shape[0], dtype='int')
    out, params = vdm.init_with_output({"sample": rng, "params": rng}, batch, conditioning)
    print(f"Params: {param_count(params):,}")

    def loss(params, rng, im, lb, beta=1.0):
        l1, l2, l3 = vdm.apply(params, im.astype('float'), lb, rngs={"sample": rng})
        rescale_to_bpd = 1./(onp.prod(im.shape[1:]) * np.log(2.0))
        return (l1.mean()/beta + l2.mean()/beta + l3.mean()) * rescale_to_bpd

    from typing import Any

    @flax.struct.dataclass
    class Store:
        params: np.ndarray
        state: Any
        rng: Any
        step: int = 0

    TSTEPS = 1_500 // jax.device_count()

    # we'll use adamw with some linear warmup and a cosine decay.
    opt = optax.chain(
        optax.scale_by_schedule(
            optax.cosine_decay_schedule(1.0, TSTEPS, 1e-5)),
        optax.adamw(8e-4, b1=0.9, b2=0.99, eps=1e-8, weight_decay=1e-4),
        optax.scale_by_schedule(
            optax.linear_schedule(0.0, 1.0, 250)))

    store = Store(params, opt.init(params), rng, 0)
    pstore = replicate(store)

    import functools

    @functools.partial(jax.pmap, axis_name='batch')
    def step(store, batch):
        rng, spl = random.split(store.rng)
        im, lb = batch
        out, grads = jax.value_and_grad(loss)(store.params, spl, im, lb)
        grads = jax.lax.pmean(grads, 'batch')
        updates, state = opt.update(grads, store.state, store.params)
        params = optax.apply_updates(store.params, updates)
        return (store.replace(
            params=params,
            state=state,
            rng=rng,
            step=store.step + 1),
            jax.lax.pmean(out, 'batch'))

    vals = []
    batches = create_input_iter(train_ds)
    ebatches = create_input_iter(test_ds)
    with trange(TSTEPS) as t:
        for i in t:
            pstore, val = step(pstore, next(batches))
            v = unreplicate(val)
            t.set_postfix(val=v)
            vals.append(v)

    plt.plot(vals[1000:])
    batch = unreplicate(next(ebatches))
    ims, lbs = batch
    loss_diff, loss_klz, loss_recon = vdm.apply(
        unreplicate(pstore.params), ims, lbs, rngs={"sample": rng})
    losses = jax.tree_map(lambda x: np.mean(x) / (onp.prod(ims.shape[1:]) * np.log(
        2)), {"loss_diff": loss_diff, "loss_klz": loss_klz, "loss_recon": loss_recon})
    print(losses, "\n", sum(losses.values()))

    loss_diff, loss_klz, loss_recon = vdm.apply(unreplicate(
        pstore.params), ims, 0 * lbs, rngs={"sample": rng})
    losses = jax.tree_map(lambda x: np.mean(x) / (onp.prod(ims.shape[1:]) * np.log(
        2)), {"loss_diff": loss_diff, "loss_klz": loss_klz, "loss_recon": loss_recon})
    print(losses, "\n", sum(losses.values()))


if __name__ == "__main__":
    main()
    pass
