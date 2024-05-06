from typing import Callable, Optional, List

import copy
import math
import pathlib
from pathlib import Path
from random import random
from functools import partial

import mindspore as ms
from mindspore import nn, ops

import torchvision.transforms as T

from beartype import beartype

from videogvt.models.vqvae import VQVAE3D
from videogvt.models.transformer.attention import FeedForward, CrossAttention
from videogvt.models.transformer.t5 import (
    get_encoded_dim,
    DEFAULT_T5_NAME,
    TextEncoder,
)

from tqdm.auto import tqdm

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def l2norm(t):
    return ops.normalize(t, dim=-1)


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


# classes


# class LayerNorm(nn.Cell):
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = ms.Parameter(ops.ones(dim))
#         self.beta = ms.Parameter(ops.zeros(dim), requires_grad=False)
#         # self.register_buffer("beta", ops.zeros(dim))

#     def construct(self, x):
#         return ops.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# class GEGLU(nn.Cell):
#     """https://arxiv.org/abs/2002.05202"""

#     def construct(self, x):
#         x, gate = x.chunk(2, dim=-1)
#         return gate * ops.gelu(x)


# def FeedForward(dim, mult=4):
#     """https://arxiv.org/abs/2110.09456"""

#     inner_dim = int(dim * mult * 2 / 3)
#     return nn.Sequential(
#         LayerNorm(dim),
#         nn.Dense(dim, inner_dim * 2, has_bias=False),
#         GEGLU(),
#         LayerNorm(inner_dim),
#         nn.Dense(inner_dim, dim, has_bias=False),
#     )


# class Attention(nn.Cell):
#     def __init__(
#         self,
#         dim,
#         dim_head=64,
#         heads=8,
#         cross_attend=False,
#         scale=8,
#         flash=True,
#         dropout=0.0,
#     ):
#         super().__init__()
#         self.scale = scale
#         self.heads = heads
#         inner_dim = dim_head * heads

#         self.cross_attend = cross_attend
#         self.norm = LayerNorm(dim)

#         self.attend = Attend(flash=flash, dropout=dropout, scale=scale)

#         self.null_kv = ms.Parameter(ops.randn(2, heads, 1, dim_head))

#         self.to_q = nn.Dense(dim, inner_dim, has_bias=False)
#         self.to_kv = nn.Dense(dim, inner_dim * 2, has_bias=False)

#         self.q_scale = ms.Parameter(ops.ones(dim_head))
#         self.k_scale = ms.Parameter(ops.ones(dim_head))

#         self.to_out = nn.Dense(inner_dim, dim, has_bias=False)

#     def construct(self, x, context=None, context_mask=None):
#         assert not (exists(context) ^ self.cross_attend)

#         n = x.shape[-2]
#         h, is_cross_attn = self.heads, exists(context)

#         x = self.norm(x)

#         kv_input = context if self.cross_attend else x

#         q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, axis=-1))

#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

#         nk, nv = self.null_kv
#         nk, nv = map(lambda t: repeat(t, "h 1 d -> b h 1 d", b=x.shape[0]), (nk, nv))

#         k = ops.cat((nk, k), axis=-2)
#         v = ops.cat((nv, v), axis=-2)

#         q, k = map(l2norm, (q, k))
#         q = q * self.q_scale
#         k = k * self.k_scale

#         if exists(context_mask):
#             context_mask = repeat(context_mask, "b j -> b h i j", h=h, i=n)
#             context_mask = ops.pad(context_mask, (1, 0), value=True)

#         out = self.attend(q, k, v, mask=context_mask)

#         out = rearrange(out, "b h n d -> b n (h d)")
#         return self.to_out(out)


class TransformerBlocks(nn.Cell):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4, flash=True):
        super().__init__()
        self.layers = nn.CellList([])

        for _ in range(depth):
            self.layers.append(
                nn.CellList(
                    [
                        CrossAttention(
                            query_dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            enable_flash_attention=flash,
                        ),
                        CrossAttention(
                            query_dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            cross_attend=True,
                            enable_flash_attention=flash,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm([dim], epsilon=1e-05)

    def construct(self, x, context=None, context_mask=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x

            x = cross_attn(x, context=context, mask=context_mask) + x

            x = ff(x) + x

        return self.norm(x)


# transformer - it's all we need


class Transformer(nn.Cell):
    def __init__(
        self,
        num_tokens,
        dim,
        seq_len,
        dim_out=None,
        t5_name=DEFAULT_T5_NAME,
        self_cond=False,
        add_mask_id=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len

        self.transformer_blocks = TransformerBlocks(dim=dim, **kwargs)
        self.norm = nn.LayerNorm([dim], epsilon=1e-05)

        self.dim_out = default(dim_out, num_tokens)
        self.to_logits = nn.Dense(dim, self.dim_out, has_bias=False)

        # text conditioning
        self.text_encoder = TextEncoder(t5_name)

        text_embed_dim = get_encoded_dim(t5_name)

        self.text_embed_proj = (
            nn.Dense(text_embed_dim, dim, has_bias=False)
            if text_embed_dim != dim
            else nn.Identity()
        )

        # optional self conditioning

        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    def encode_text(self, texts):
        return self.text_encoder.encode(texts)

    def forward_with_cond_scale(
        self, *args, cond_scale=3.0, return_embed=False, **kwargs
    ):
        if cond_scale == 1:
            return self.construct(
                *args, return_embed=return_embed, cond_drop_prob=0.0, **kwargs
            )

        logits, embed = self.construct(
            *args, return_embed=True, cond_drop_prob=0.0, **kwargs
        )

        null_logits = self.construct(*args, cond_drop_prob=1.0, **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        *args,
        text_embed: ms.Tensor,
        neg_text_embed: ms.Tensor,
        cond_scale=3.0,
        return_embed=False,
        **kwargs
    ):
        neg_logits = self.construct(
            *args, neg_text_embed=neg_text_embed, cond_drop_prob=0.0, **kwargs
        )
        pos_logits, embed = self.construct(
            *args,
            return_embed=True,
            text_embed=text_embed,
            cond_drop_prob=0.0,
            **kwargs
        )

        logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return logits, embed

        return logits

    def construct(
        self,
        x,
        texts: List[str],
        return_embed=False,
        return_logits=False,
        labels=None,
        ignore_index=0,
        self_cond_embed=None,
        cond_drop_prob=0.0,
        conditioning_token_ids: Optional[ms.Tensor] = None,
    ):
        b, n = x.shape
        assert not self.seq_len < n

        # prepare texts

        text_embeds = self.text_encoder.encode(texts)

        context = self.text_embed_proj(text_embeds)

        context_mask = (text_embeds != 0).any(axis=-1)

        # classifier free guidance

        if cond_drop_prob > 0.0:
            mask = prob_mask_like((b, 1), 1.0 - cond_drop_prob)
            context_mask = context_mask & mask

        # concat conditioning video token ids if needed

        if exists(conditioning_token_ids):
            # conditioning_token_ids = rearrange(
            #     conditioning_token_ids, "b ... -> b (...)"
            # )
            conditioning_token_ids = conditioning_token_ids.reshape(conditioning_token_ids.shape[0], -1)
            cond_token_emb = self.token_emb(conditioning_token_ids)
            context = ops.cat((context, cond_token_emb), axis=-2)
            context_mask = ops.pad(
                context_mask, (0, conditioning_token_ids.shape[-1]), value=True
            )

        # embed tokens

        x = self.token_emb(x)
        x = x + self.pos_emb(ops.arange(n))

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = ops.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(x, context=context, context_mask=context_mask)

        logits = self.to_logits(embed)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            # loss = ops.binary_cross_entropy_with_logits(
            #     rearrange(logits, "... 1 -> ..."), labels
            # )
            loss = ops.binary_cross_entropy_with_logits(
                logits.squeeze(-1), labels
            )
        else:
            # loss = ops.cross_entropy(
            #     rearrange(logits, "b n c -> b c n"), labels, ignore_index=ignore_index
            # )
            loss = ops.cross_entropy(
                logits.swapaxes(1, 2), labels, ignore_index=ignore_index
            )

        if not return_logits:
            return loss

        return loss, logits


# self critic wrapper


class SelfCritic(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Dense(net.dim, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(
            x, *args, return_embed=True, **kwargs
        )
        return self.to_pred(embeds)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(
            x, *args, return_embed=True, **kwargs
        )
        return self.to_pred(embeds)

    def construct(self, x, *args, labels=None, **kwargs):
        _, embeds = self.net(x, *args, return_embed=True, **kwargs)
        logits = self.to_pred(embeds)

        if not exists(labels):
            return logits

        # logits = rearrange(logits, "... 1 -> ...")
        logits = logits.squeeze(-1)
        return ops.binary_cross_entropy_with_logits(logits, labels)


# specialized transformers


class MAGVITransformer(Transformer):
    def __init__(self, *args, **kwargs):
        assert "add_mask_id" not in kwargs
        super().__init__(*args, add_mask_id=True, **kwargs)


class TokenCritic(Transformer):
    def __init__(self, *args, **kwargs):
        assert "dim_out" not in kwargs
        super().__init__(*args, dim_out=1, **kwargs)


# classifier free guidance functions


def uniform(shape, min=0, max=1):
    return ops.zeros(shape).float().uniform_(0, 1)


@ms.jit
def prob_mask_like(shape, prob):
    if prob == 1:
        return ops.ones(shape, dtype=ms.bool)
    elif prob == 0:
        return ops.zeros(shape, dtype=ms.bool)
    else:
        return uniform(shape) < prob


# sampling helpers


def log(t, eps=1e-20):
    return ops.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = ops.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = ops.full_like(logits, float("-inf"))
    probs = probs.scatter(2, ind, val)
    return probs


# noise schedules


def cosine_schedule(t):
    return ops.cos(t * math.pi * 0.5)


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

        if exists(vae):
            self.vae = copy.deepcopy(vae)
            self.vae.set_train(False)
        else:
            self.vae = None

        if exists(cond_vae):
            self.cond_vae = cond_vae.set_train(False)
        else:
            self.cond_vae = self.vae

        assert not (
            exists(cond_vae) and not exists(cond_video_size)
        ), "cond_video_size must be specified if conditioning"

        self.video_size = video_size
        self.cond_video_size = cond_video_size
        self.resize_video_for_cond_video = exists(cond_video_size)

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

        assert not (self_token_critic and exists(token_critic))
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
                ops.linspace(0, 1, timesteps),
                reversed(range(timesteps)),
            ),
            total=timesteps,
        ):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices

            ids = ids.scatter(1, masked_indices, self.mask_id)

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
                probs_without_temperature = logits.softmax(dim=-1)

                scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
                scores = scores.squeeze(-1)
                # scores = rearrange(scores, "... 1 -> ...")

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, -1e5)
                else:
                    assert (
                        self.no_mask_token_prob > 0.0
                    ), "without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token"

        # get ids

        # ids = rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
        ids = ids.reshape(ids.shape[0], fmap_size, fmap_size)

        if not exists(self.vae):
            return ids

        videos = self.vae.decode_from_ids(ids)
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
