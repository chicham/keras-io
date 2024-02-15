"""
Title: Fine-tuning Stable Diffusion
Author: [Sayak Paul](https://twitter.com/RisingSayak), [Chansung Park](https://twitter.com/algo_diver)
Date created: 2022/12/28
Last modified: 2023/01/13
Description: Fine-tuning Stable Diffusion using a custom image-caption dataset.
Accelerator: GPU
"""

"""
## Introduction

This tutorial shows how to fine-tune a
[Stable Diffusion model](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/)
on a custom dataset of `{image, caption}` pairs. We build on top of the fine-tuning
script provided by Hugging Face
[here](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).

We assume that you have a high-level understanding of the Stable Diffusion model.
The following resources can be helpful if you're looking for more information in that regard:

* [High-performance image generation using Stable Diffusion in KerasCV](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/)
* [Stable Diffusion with Diffusers](https://huggingface.co/blog/stable_diffusion)

It's highly recommended that you use a GPU with at least 30GB of memory to execute
the code.

By the end of the guide, you'll be able to generate images of interesting Pokémon:

![custom-pokemons](https://i.imgur.com/X4m614M.png)

The tutorial relies on KerasCV 0.4.0. Additionally, we need
at least TensorFlow 2.11 in order to use AdamW with mixed precision.
"""

"""shell
pip uninstall -y tensorflow tensorboard
# pip install keras==3.0.4 keras-cv==0.8.2 jax[cuda12_pip]==0.4.24 tf-nightly-cpu==2.16.0.dev20240101 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -q keras==3.0.4 keras-cv==0.8.2 tf-nightly[and-cuda]==2.16.0.dev20240101 tb-nightly==2.16.0a20240101 tensorboard_plugin_profile
"""

"""
## What are we fine-tuning?

A Stable Diffusion model can be decomposed into several key models:

* A text encoder that projects the input prompt to a latent space. (The caption
associated with an image is referred to as the "prompt".)
* A variational autoencoder (VAE) that projects an input image to a latent space acting
as an image vector space.
* A diffusion model that refines a latent vector and produces another latent vector, conditioned
on the encoded text prompt
* A decoder that generates images given a latent vector from the diffusion model.

It's worth noting that during the process of generating an image from a text prompt, the
image encoder is not typically employed.

However, during the process of fine-tuning, the workflow goes like the following:

1. An input text prompt is projected to a latent space by the text encoder.
2. An input image is projected to a latent space by the image encoder portion of the VAE.
3. A small amount of noise is added to the image latent vector for a given timestep.
4. The diffusion model uses latent vectors from these two spaces along with a timestep embedding
to predict the noise that was added to the image latent.
5. A reconstruction loss is calculated between the predicted noise and the original noise
added in step 3.
6. Finally, the diffusion model parameters are optimized w.r.t this loss using
gradient descent.

Note that only the diffusion model parameters are updated during fine-tuning, while the
(pre-trained) text and the image encoders are kept frozen.

Don't worry if this sounds complicated. The code is much simpler than this!
"""

"""
## Imports
"""

import os
from textwrap import wrap

import keras
import matplotlib.pyplot as plt
import pandas as pd
from keras_cv import models as models_cv
from keras import ops, layers, utils, models
from keras import random
import itertools
import math

# %load_ext tensorboard

"""
## Data loading

We use the dataset
[Pokémon BLIP captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).
However, we'll use a slightly different version which was derived from the original
dataset to fit better with `tf.data`. Refer to
[the documentation](https://huggingface.co/datasets/sayakpaul/pokemon-blip-original-version)
for more details.
"""

data_path = utils.get_file(
    origin="https://huggingface.co/datasets/sayakpaul/pokemon-blip-original-version/resolve/main/pokemon_dataset.tar.gz",
    untar=True,
)

data_frame = pd.read_csv(os.path.join(data_path, "data.csv"))

data_frame["image_path"] = data_frame["image_path"].apply(
    lambda x: os.path.join(data_path, x)
)
data_frame.head()

"""
Since we have only 833 `{image, caption}` pairs, we can precompute the text embeddings from
the captions. Moreover, the text encoder will be kept frozen during the course of
fine-tuning, so we can save some compute by doing this.

Before we use the text encoder, we need to tokenize the captions.
"""


"""
## Prepare a `tf.data.Dataset`

In this section, we'll prepare a `tf.data.Dataset` object from the input image file paths
and their corresponding caption tokens. The section will include the following:

* Pre-computation of the text embeddings from the tokenized captions.
* Loading and augmentation of the input images.
* Shuffling and batching of the dataset.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

RESOLUTION = 256
BATCH_SIZE = 32
# N_EPOCHS = 100
N_EPOCHS = 50

MAX_PROMPT_LENGTH = 77
SEED = 42
USE_MP = True
POS_IDS = ops.expand_dims(ops.arange(MAX_PROMPT_LENGTH, dtype="int32"), 0)

CKPT_PATH = "finetuned_stable_diffusion.weights.h5"

if USE_MP:
    keras.mixed_precision.set_global_policy("mixed_float16")

augmenter = models.Sequential(
    layers=[
        layers.CenterCrop(RESOLUTION, RESOLUTION),
        layers.RandomFlip(seed=SEED),
        layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)


def get_timestep_embeddings(timesteps, dim=320, max_period=10000, dtype="float32"):
    half = dim // 2
    span = ops.cast(ops.arange(0, half), dtype=dtype)
    span = ops.reshape(span, (1, -1))
    max_period = ops.cast(max_period, dtype=span.dtype)
    freqs = ops.exp(-ops.log(max_period) * span / half)
    timesteps = ops.cast(timesteps, dtype=dtype)
    timesteps = ops.reshape(timesteps, (-1, 1))
    args = timesteps * freqs
    embeddings = ops.concatenate([ops.cos(args), ops.sin(args)], axis=1)
    return embeddings


def sample_normal(embeddings, seed=None):
    mean, logvar = ops.split(embeddings, 2, axis=-1)
    logvar = ops.clip(logvar, -30, 20)
    std = ops.exp(0.5 * logvar)
    sample = random.normal(ops.shape(mean), dtype=mean.dtype, seed=seed)
    return mean + std * sample


class PokemonBlipDataset(keras.utils.PyDataset):
    def __init__(
        self,
        captions: list[str],
        image_paths: list[str],
        batch_size: int,
        workers: int = 1,
        tokenizer=None,
        max_prompt_length: int = MAX_PROMPT_LENGTH,
        resolution: int = RESOLUTION,
        seed: int = 42,
        use_multiprocessing: bool = False,
        max_queue_size: int = 10,
        end_of_text_token: int = None,
    ):
        super().__init__(
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
        )
        self.captions = ops.convert_to_numpy(captions)
        self.image_paths = ops.convert_to_numpy(image_paths)
        self.batch_size = batch_size
        self.max_prompt_length = max_prompt_length
        self.resolution = resolution

        if tokenizer is None:
            tokenizer = models_cv.stable_diffusion.SimpleTokenizer()

        self.tokenizer = tokenizer

        self.seed = seed

        indices = ops.arange(len(captions))
        self.indices = random.shuffle(indices, seed=self.seed)
        self.end_of_text_token = end_of_text_token

    def transform_fn(self, image_path, caption):
        image = utils.load_img(
            image_path, target_size=(self.resolution, self.resolution)
        )
        image = ops.convert_to_tensor(image)
        tokens = self.tokenizer.encode(caption)
        if len(tokens) > self.max_prompt_length:
            raise ValueError(
                f"Prompt is too long (should be <= {self.max_prompt_length} tokens)"
            )
        tokens = ops.convert_to_tensor(
            [tokens + [self.end_of_text_token] * (self.max_prompt_length - len(tokens))]
        )

        return image, tokens

    def __getitem__(self, idx):
        # Return x, y for batch idx.
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.indices))
        indices = self.indices[low:high]

        # TODO: do not use keras for shuffling but python function
        batch_captions = self.captions[indices]
        batch_image_paths = self.image_paths[indices]

        batch = (
            self.transform_fn(img, caption)
            for caption, img in zip(batch_captions, batch_image_paths)
        )
        batch_images, batch_tokens = zip(*batch)

        batch_tokens = ops.concatenate(batch_tokens, axis=0)

        batch_images = augmenter(batch_images)

        targets = keras.random.normal(
            (self.batch_size, self.resolution // 8, self.resolution // 8, 4),
            seed=self.seed,
            dtype=batch_images.dtype,
        )

        inputs = {
            "image": batch_images,
            "token": batch_tokens,
            "noise": targets,
        }
        return inputs, targets

    def __len__(self):
        # Return number of batches.
        return math.ceil(len(self.indices) / self.batch_size)

    def on_epoch_end(self):
        self.indices = random.shuffle(self.indices, seed=self.seed)


"""
The baseline Stable Diffusion model was trained using images with 512x512 resolution. It's
unlikely for a model that's trained using higher-resolution images to transfer well to
lower-resolution images. However, the current model will lead to OOM if we keep the
resolution to 512x512 (without enabling mixed-precision). Therefore, in the interest of
interactive demonstrations, we kept the input resolution to 256x256.
"""

# Prepare the dataset.
tokenizer = models_cv.stable_diffusion.SimpleTokenizer()

all_captions = data_frame["caption"].to_list()
all_image_paths = data_frame["image_path"].to_list()
del data_frame

training_dataset = PokemonBlipDataset(
    captions=all_captions,
    image_paths=all_image_paths,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    use_multiprocessing=False,
    resolution=RESOLUTION,
    max_prompt_length=MAX_PROMPT_LENGTH,
    end_of_text_token=tokenizer.end_of_text,
)
# Take a sample batch and investigate.
sample_batch, _ = training_dataset[0]

for k, v in sample_batch.items():
    print(k, ops.shape(v))

"""
We can also take a look at the training images and their corresponding captions.
"""

plt.figure(figsize=(20, 10))

for i in range(3):
    ax = plt.subplot(1, 4, i + 1)
    img = ops.cast(sample_batch["image"][i], dtype="float32")
    plt.imshow((img + 1.0) / 2)

    tokens = ops.convert_to_numpy(sample_batch["token"][i])
    text = tokenizer.decode(tokens)
    text = text.replace("<|startoftext|>", "")
    text = text.replace("<|endoftext|>", "")
    text = "\n".join(wrap(text, 12))

    plt.title(text, fontsize=15)
    plt.axis("off")

"""
## A trainer class for the fine-tuning loop
"""


class StableDiffusionTrainer(keras.Model):
    def __init__(
        self,
        diffusion_model: keras.Model,
        noise_scheduler: keras.Model,
        text_encoder: keras.Model,
        vae: keras.Model,
        seed: int = None,
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        noise_scheduler.trainable = False
        self.noise_scheduler = noise_scheduler
        text_encoder.trainable = False
        self.text_encoder = text_encoder
        vae.trainable = False
        self.vae = vae
        self.seed = seed

    def call(self, inputs):
        batch_size = ops.shape(inputs["token"])[0]
        pos_ids = ops.repeat(POS_IDS, batch_size, axis=0)
        contexts = self.text_encoder([inputs["token"], pos_ids])
        embeddings = self.vae(inputs["image"])
        latents = sample_normal(embeddings, seed=self.seed)
        latents = latents * 0.18215

        timesteps = keras.random.randint(
            (batch_size,),
            0,
            self.noise_scheduler.train_timesteps,
            seed=self.seed,
        )
        timestep_embeddings = get_timestep_embeddings(timesteps, dtype=latents.dtype)
        noisy_latents = self.noise_scheduler.add_noise(
            latents, inputs["noise"], timesteps
        )
        preds = self.diffusion_model(
            {
                "context": contexts,
                "latent": noisy_latents,
                "timestep_embedding": timestep_embeddings,
            }
        )
        return preds

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Overriding this method will allow us to use the `ModelCheckpoint`
        # callback directly with this trainer class. In this case, it will
        # only checkpoint the `diffusion_model` since that's what we're training
        # during fine-tuning.
        self.diffusion_model.save_weights(
            filepath=filepath,
            overwrite=overwrite,
        )


"""
One important implementation detail to note here: Instead of directly taking
the latent vector produced by the image encoder (which is a VAE), we sample from the
mean and log-variance predicted by it. This way, we can achieve better sample
quality and diversity.

It's common to add support for mixed-precision training along with exponential
moving averaging of model weights for fine-tuning these models. However, in the interest
of brevity, we discard those elements. More on this later in the tutorial.
"""

"""
## Initialize the trainer and compile it
"""

# Enable mixed-precision training if the underlying GPU has tensor cores.

diffusion_model = models_cv.stable_diffusion.DiffusionModel(
    RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH
)
text_encoder = models_cv.stable_diffusion.TextEncoder(MAX_PROMPT_LENGTH)
image_encoder = models_cv.stable_diffusion.ImageEncoder()

vae = keras.Sequential(image_encoder.layers[:-1])
diffusion_trainer = StableDiffusionTrainer(
    diffusion_model=diffusion_model,
    noise_scheduler=models_cv.stable_diffusion.NoiseScheduler(),
    text_encoder=text_encoder,
    vae=vae,
)

# These hyperparameters come from this tutorial by Hugging Face:
# https://huggingface.co/docs/diffusers/training/text2image
lr = 1e-5
beta_1, beta_2 = 0.9, 0.999
weight_decay = 1e-2
epsilon = 1e-08
clipnorm = 1.0
use_ema = True
ema_momentum = 0.9999

optimizer = keras.optimizers.AdamW(
    learning_rate=lr,
    weight_decay=weight_decay,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
    clipnorm=clipnorm,
    use_ema=use_ema,
)

diffusion_trainer.compile(
    optimizer=optimizer, loss="mse", auto_scale_loss=True, jit_compile="auto"
)

"""
## Fine-tuning

To keep the runtime of this tutorial short, we just fine-tune for an epoch.
"""

# %tensorboard --logdir ./logs


ckpt_callback = keras.callbacks.ModelCheckpoint(
    CKPT_PATH,
    save_weights_only=True,
    monitor="loss",
    mode="min",
)
profile_batch = (20, 50) if keras.backend.backend() == "tensorflow" else None
diffusion_trainer.fit(
    training_dataset,
    epochs=N_EPOCHS,
    callbacks=[
        ckpt_callback,
        keras.callbacks.TensorBoard(log_dir="./logs", profile_batch=(20, 50)),
    ],
)

"""
## Inference

We fine-tuned the model for 60 epochs on an image resolution of 512x512. To allow
training with this resolution, we incorporated mixed-precision support. You can
check out
[this repository](https://github.com/sayakpaul/stabe-diffusion-keras-ft)
for more details. It additionally provides support for exponential moving averaging of
the fine-tuned model parameters and model checkpointing.


For this section, we'll use the checkpoint derived after 60 epochs of fine-tuning.
"""

# TODO: Finetune the model then upload the checkpoint
# weights_path = keras.utils.get_file(
#     origin="https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned/resolve/main/ckpt_epochs_72_res_512_mp_True.h5"
# )

img_height = img_width = RESOLUTION
pokemon_model = models_cv.StableDiffusion(img_width=img_width, img_height=img_height)
# We just reload the weights of the fine-tuned diffusion model.
pokemon_model.diffusion_model.load_weights(CKPT_PATH)

"""
Now, we can take this model for a test-drive.
"""

prompts = ["Yoda", "Hello Kitty", "A pokemon with red eyes"]
images_to_generate = 3
outputs = {}

for prompt in prompts:
    generated_images = pokemon_model.text_to_image(
        prompt, batch_size=images_to_generate, unconditional_guidance_scale=40
    )
    outputs.update({prompt: generated_images})

"""
With 60 epochs of fine-tuning (a good number is about 70), the generated images were not
up to the mark. So, we experimented with the number of steps Stable Diffusion takes
during the inference time and the `unconditional_guidance_scale` parameter.

We found the best results with this checkpoint with `unconditional_guidance_scale` set to
40.
"""


def plot_images(images, title):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        _ = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(title, fontsize=12)
        plt.axis("off")


for prompt in outputs:
    plot_images(outputs[prompt], prompt)

"""
We can notice that the model has started adapting to the style of our dataset. You can
check the
[accompanying repository](https://github.com/sayakpaul/stable-diffusion-keras-ft#results)
for more comparisons and commentary. If you're feeling adventurous to try out a demo,
you can check out
[this resource](https://huggingface.co/spaces/sayakpaul/pokemon-sd-kerascv).
"""

"""
## Conclusion and acknowledgements

We demonstrated how to fine-tune the Stable Diffusion model on a custom dataset. While
the results are far from aesthetically pleasing, we believe with more epochs of
fine-tuning, they will likely improve. To enable that, having support for gradient
accumulation and distributed training is crucial. This can be thought of as the next step
in this tutorial.

There is another interesting way in which Stable Diffusion models can be fine-tuned,
called textual inversion. You can refer to
[this tutorial](https://keras.io/examples/generative/fine_tune_via_textual_inversion/)
to know more about it.

We'd like to acknowledge the GCP Credit support from ML Developer Programs' team at
Google. We'd like to thank the Hugging Face team for providing the
[fine-tuning script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)
. It's very readable and easy to understand.
"""
