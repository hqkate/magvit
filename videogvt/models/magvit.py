from typing import Callable, Optional, List

import copy
import math
from pathlib import Path
from random import random
from functools import partial

from beartype import beartype
from tqdm.auto import tqdm

import numpy as np
import mindspore as ms
from mindspore import nn, ops
from videogvt.models.vqvae import VQVAE3D
from videogvt.models.transformer.transformer import (
    MAGVITransformer,
    TokenCritic,
    SelfCritic,
    exists,
    default,
    uniform,
)


# helpers


def log(t, eps=1e-20):
    return ops.log(t.clamp(min=eps))


def gumbel_noise(t):
    # noise = ops.zeros_like(t).uniform_(0, 1)
    noise = ops.uniform(t.shape, ms.Tensor(0.0), ms.Tensor(1.0))
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(axis=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = ops.full_like(logits, float("-inf"))
    probs = probs.scatter(2, ind, val)
    return probs


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.set_train(False)
        out = fn(model, *args, **kwargs)
        model.set_train(was_training)
        return out

    return inner


# tensor helpers


def get_mask_subset_prob(mask, prob, min_mask=0):
    batch, seq = mask.shape
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = ops.rand((batch, seq))
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim=-1).argsort(dim=-1).float()

    num_padding = (~mask).sum(dim=-1, keepdim=True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask


# noise schedules


def cosine_schedule(t):
    return math.cos(t * math.pi * 0.5)


# main maskgit classes


@beartype
class MAGVIT(nn.Cell):
    def __init__(
        self,
        video_size,
        transformer: MAGVITransformer,
        noise_schedule: Callable = cosine_schedule,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic=False,
        vae: Optional[VQVAE3D] = None,
        cond_vae: Optional[VQVAE3D] = None,
        cond_video_size=None,
        cond_drop_prob=0.5,
        self_cond_prob=0.9,
        no_mask_token_prob=0.0,
        critic_loss_weight=1.0,
    ):
        super().__init__()

        if vae is not None:
            self.vae = copy.deepcopy(vae)
            self.vae.set_train(False)
        else:
            self.vae = None

        if cond_vae is not None:
            self.cond_vae = cond_vae.set_train(False)
        else:
            self.cond_vae = self.vae

        assert not (
            (cond_vae is not None) and not (cond_video_size is not None)
        ), "cond_video_size must be specified if conditioning"

        self.video_size = video_size
        self.cond_video_size = cond_video_size
        self.resize_video_for_cond_video = cond_video_size is not None

        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.self_cond = transformer.self_cond
        assert (
            self.vae.codebook_size
            == self.cond_vae.codebook_size
            == transformer.num_tokens
        ), "transformer num_tokens must be set to be equal to the vae codebook size"

        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

        assert not (self_token_critic and (token_critic is not None))
        self.token_critic = token_critic

        if self_token_critic:
            self.token_critic = SelfCritic(transformer)

        self.critic_loss_weight = critic_loss_weight

        # self conditioning
        self.self_cond_prob = self_cond_prob

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        self.no_mask_token_prob = no_mask_token_prob

    def save(self, path):
        ms.save_checkpoint(self, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self = ms.load_checkpoint(str(path))

    @eval_decorator
    def generate(
        self,
        texts: List[str],
        negative_texts: Optional[List[str]] = None,
        cond_videos: Optional[ms.Tensor] = None,
        fmap_size=None,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        force_not_use_token_critic=False,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        cond_scale=3,
        critic_noise_scale=1,
    ):
        fmap_size = default(fmap_size, self.vae.get_encoded_fmap_size(self.video_size))

        # begin with all video token ids masked

        seq_len = fmap_size**2

        batch_size = len(texts)

        shape = (batch_size, seq_len)

        ids = ops.full(shape, self.mask_id, dtype=ms.int32)
        scores = ops.zeros(shape, dtype=ms.float32)

        starting_temperature = temperature

        cond_ids = None

        text_embeds = self.transformer.encode_text(texts)

        demask_fn = self.transformer.forward_with_cond_scale

        # whether to use token critic for scores

        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        # negative prompting, as in paper

        neg_text_embeds = None
        if exists(negative_texts):
            assert len(texts) == len(negative_texts)

            neg_text_embeds = self.transformer.encode_text(negative_texts)
            demask_fn = partial(
                self.transformer.forward_with_neg_prompt,
                neg_text_embeds=neg_text_embeds,
            )

            if use_token_critic:
                token_critic_fn = partial(
                    self.token_critic.forward_with_neg_prompt,
                    neg_text_embeds=neg_text_embeds,
                )

        if self.resize_video_for_cond_video:
            assert exists(
                cond_videos
            ), "conditioning video must be passed in to generate for super res maskgit"

            _, cond_ids, _ = ops.stop_gradient(self.cond_vae.encode(cond_videos))

        self_cond_embed = None

        for timestep, steps_until_x0 in tqdm(
            zip(
                np.linspace(0, 1, timesteps),
                reversed(range(timesteps)),
            ),
            total=timesteps,
        ):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = int(max((rand_mask_prob * seq_len), 1))

            _, masked_indices = scores.topk(num_token_masked, dim=-1)

            src_ids = ops.full_like(masked_indices, self.mask_id)
            ids = ops.scatter(ids, axis=1, index=masked_indices, src=src_ids)

            logits, embed = demask_fn(
                ids,
                text_embeds=text_embeds,
                self_cond_embed=self_cond_embed,
                conditioning_token_ids=cond_ids,
                cond_scale=cond_scale,
                return_embed=True,
            )

            self_cond_embed = embed if self.self_cond else None

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            pred_ids = pred_ids.astype(ms.int32)
            is_mask = ids == self.mask_id

            ids = ops.where(is_mask, pred_ids, ids)

            if use_token_critic:
                scores = token_critic_fn(
                    ids,
                    text_embeds=text_embeds,
                    conditioning_token_ids=cond_ids,
                    cond_scale=cond_scale,
                )

                # scores = rearrange(scores, "... 1 -> ...")
                scores = scores.squeeze(-1)
                scores = scores + (uniform(scores.shape) - 0.5) * critic_noise_scale * (
                    steps_until_x0 / timesteps
                )

            else:
                bs = logits.shape[0]
                probs_without_temperature = ops.softmax(logits, axis=-1)
                scores = 1 - probs_without_temperature.gather(
                    pred_ids[..., None], axis=2, batch_dims=bs
                )
                scores = scores.squeeze(-1)
                # scores = rearrange(scores, "... 1 -> ...")

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, ms.Tensor(-1e5))
                else:
                    assert (
                        self.no_mask_token_prob > 0.0
                    ), "without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token"

        # get ids

        # ids = rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
        ids = ids.reshape(ids.shape[0], fmap_size, fmap_size)

        if not exists(self.vae):
            return ids

        videos = self.vae.decode_from_indices(ids)
        return videos

    def construct(
        self,
        videos_or_ids: ms.Tensor,
        ignore_index=-1,
        cond_videos: Optional[ms.Tensor] = None,
        cond_token_ids: Optional[ms.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[ms.Tensor] = None,
        cond_drop_prob=None,
        train_only_generator=False,
        sample_temperature=None,
    ):
        # tokenize if needed

        if videos_or_ids.dtype == ms.float:
            assert exists(
                self.vae
            ), "vqgan vae must be passed in if training from raw videos"
            assert all(
                [
                    height_or_width == self.video_size
                    for height_or_width in videos_or_ids.shape[-2:]
                ]
            ), "the video you passed in is not of the correct dimensions"

            _, ids, _ = ops.stop_gradient(self.vae.encode(videos_or_ids))
        else:
            assert (
                not self.resize_video_for_cond_video
            ), "you cannot pass in raw video token ids if you want the framework to autoresize video for conditioning super res transformer"
            ids = videos_or_ids

        # take care of conditioning video if specified

        if self.resize_video_for_cond_video:
            cond_videos_or_ids = ops.interpolate(
                videos_or_ids, self.cond_video_size, mode="nearest"
            )

        # get some basic variables

        # ids = rearrange(ids, "b ... -> b (...)")
        ids = ids.reshape(ids.shape[0], -1)

        batch, seq_len, cond_drop_prob = (
            *ids.shape,
            default(cond_drop_prob, self.cond_drop_prob),
        )

        # tokenize conditional videos if needed

        assert not (
            exists(cond_videos) and exists(cond_token_ids)
        ), "if conditioning on low resolution, cannot pass in both videos and token ids"

        if exists(cond_videos):
            assert exists(self.cond_vae), "cond vqgan vae must be passed in"
            assert all(
                [
                    height_or_width == self.cond_video_size
                    for height_or_width in cond_videos.shape[-2:]
                ]
            )

            _, cond_token_ids, _ = ops.stop_gradient(self.cond_vae.encode(cond_videos))

        # prepare mask

        rand_time = uniform((batch,))
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

        mask_id = self.mask_id
        batch_randperm = ops.rand((batch, seq_len)).argsort(dim=-1)
        # mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")
        mask = batch_randperm < num_token_masked.unsqueeze(-1)

        mask_id = self.transformer.mask_id
        labels = ops.where(mask, ids, ignore_index)

        if self.no_mask_token_prob > 0.0:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x = ops.where(mask, mask_id, ids)

        # get text embeddings

        if exists(texts):
            text_embeds = self.transformer.encode_text(texts)
            texts = None

        # self conditioning

        self_cond_embed = None

        if self.transformer.self_cond and random() < self.self_cond_prob:
            _, self_cond_embed = ops.stop_gradient(
                self.transformer(
                    x,
                    text_embeds=text_embeds,
                    conditioning_token_ids=cond_token_ids,
                    cond_drop_prob=0.0,
                    return_embed=True,
                )
            )

        # get loss

        ce_loss, logits = self.transformer(
            x,
            text_embeds=text_embeds,
            self_cond_embed=self_cond_embed,
            conditioning_token_ids=cond_token_ids,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            return_logits=True,
        )

        if not exists(self.token_critic) or train_only_generator:
            return ce_loss

        # token critic loss

        sampled_ids = gumbel_sample(
            logits, temperature=default(sample_temperature, random())
        )

        critic_input = ops.where(mask, sampled_ids, x)
        critic_labels = (ids != critic_input).float()

        bce_loss = self.token_critic(
            critic_input,
            text_embeds=text_embeds,
            conditioning_token_ids=cond_token_ids,
            labels=critic_labels,
            cond_drop_prob=cond_drop_prob,
        )

        return ce_loss + self.critic_loss_weight * bce_loss
