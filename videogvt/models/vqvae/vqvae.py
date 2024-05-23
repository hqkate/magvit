import logging

import mindspore as ms
import numpy as np
from mindspore import nn, ops

from videogvt.models.vqvae.encoder import Encoder, Encoder3D
from videogvt.models.vqvae.vector_quantizer import VQ
from videogvt.models.vqvae.lookup_free_quantization import LFQ
from videogvt.models.vqvae.decoder import Decoder, Decoder3D
from videogvt.models.vqvae.model_utils import SameConv2d, pad_at_dim, CausalConv3d


logger = logging.getLogger(__name__)


class VQVAE(nn.Cell):
    def __init__(
        self,
        h_dim,
        res_h_dim,
        n_res_layers,
        n_embeddings,
        embedding_dim,
        beta,
        save_img_embedding_map=False,
    ):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1
        )
        # pass continuous latent vector through discretization bottleneck
        self.quantizer = VQ(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def construct(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.quantizer(z_e)

        x_hat = self.decoder(z_q)

        if verbose:
            print("original data shape:", x.shape)
            print("encoded data shape:", z_e.shape)
            print("recon data shape:", x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity


class VQVAE3D(nn.Cell):
    def __init__(
        self,
        config,
        save_img_embedding_map=False,
        lookup_free_quantization=False,
        video_contains_first_frame=False,
        separate_first_frame_encoding=False,
        is_training=False,
        dtype=ms.float32,
    ):
        super(VQVAE3D, self).__init__()
        self.config = config

        # encode image into continuous latent space
        self.encoder = Encoder3D(config, dtype=dtype)
        # decode the discrete latent representation
        self.decoder = Decoder3D(config, dtype=dtype)

        encode_first_frame_separately = (
            separate_first_frame_encoding and video_contains_first_frame
        )

        embedding_dim = config.vqvae.embedding_dim
        in_dim = config.vqvae.channels
        h_dim = config.vqvae.filters
        m_dim = config.vqvae.middle_channles
        beta = config.vqvae.commitment_cost
        temporal_downsample = config.vqvae.temporal_downsample
        time_downsample_factor = 2 ** (sum(temporal_downsample) - 1)
        self.codebook_size = config.vqvae.codebook_size

        # encoder conv_in
        input_conv_kernel_size = (3, 3, 3)
        self.conv_in = CausalConv3d(in_dim, h_dim, input_conv_kernel_size, padding=1)

        # decoder conv_out
        output_conv_kernel_size = (3, 3, 3)
        self.conv_out = CausalConv3d(h_dim, in_dim, output_conv_kernel_size, padding=1)

        # whether to encode the first frame separately or not

        self.conv_in_first_frame = nn.Identity()
        self.conv_out_first_frame = nn.Identity()

        if separate_first_frame_encoding:
            self.conv_in_first_frame = SameConv2d(
                in_dim, h_dim, input_conv_kernel_size[-2:]
            )
            self.conv_out_first_frame = SameConv2d(
                h_dim, in_dim, output_conv_kernel_size[-2:]
            )

        self.separate_first_frame_encoding = separate_first_frame_encoding
        self.encode_first_frame_separately = encode_first_frame_separately
        self.video_contains_first_frame = video_contains_first_frame

        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1

        self.pre_quantization_conv = nn.Conv3d(m_dim, m_dim, kernel_size=1, stride=1, dtype=dtype)

        # pass continuous latent vector through discretization bottleneck
        if lookup_free_quantization:
            self.quantizer = LFQ(
                dim=m_dim,
                codebook_size=self.codebook_size,
                return_loss_breakdown=False,
                is_training=is_training,
                # **lfq_kwargs
            )
            logger.info("Using Lookup Free Quantization.")
        else:
            self.quantizer = VQ(self.codebook_size, embedding_dim, beta)
            logger.info("Using basic Vector Quantization.")

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(self.codebook_size)}
        else:
            self.img_to_embedding_map = None

    def init_from_vae2d(self, path):
        # default: tail init
        # path: path to vae 2d model ckpt
        vae2d_sd = ms.load_checkpoint(path)
        vae_2d_keys = list(vae2d_sd.keys())
        vae_3d_keys = list(self.parameters_dict().keys())

        # 3d -> 2d
        map_dict = {
            "conv.weight": "weight",
            "conv.bias": "bias",
        }

        new_state_dict = {}
        for key_3d in vae_3d_keys:
            if key_3d.startswith("loss"):
                continue

            # param name mapping from vae-3d to vae-2d
            key_2d = key_3d
            for kw in map_dict:
                key_2d = key_2d.replace(kw, map_dict[kw])

            assert key_2d in vae_2d_keys, f"Key {key_2d} ({key_3d}) not found in 2D VAE"

            # set vae 3d state dict
            shape_3d = self.parameters_dict()[key_3d].shape
            shape_2d = vae2d_sd[key_2d].shape
            if "bias" in key_2d:
                assert (
                    shape_3d == shape_2d
                ), f"Shape mismatch for key {key_3d} ({key_2d})"
                new_state_dict[key_3d] = vae2d_sd[key_2d]
            elif "norm" in key_2d:
                assert (
                    shape_3d == shape_2d
                ), f"Shape mismatch for key {key_3d} ({key_2d})"
                new_state_dict[key_3d] = vae2d_sd[key_2d]
            elif "conv" in key_2d or "nin_shortcut" in key_2d:
                if shape_3d[:2] != shape_2d[:2]:
                    logger.info(key_2d, shape_3d, shape_2d)
                w = vae2d_sd[key_2d]
                new_w = ms.ops.zeros(shape_3d, dtype=w.dtype)
                # tail initialization
                new_w[:, :, -1, :, :] = w  # cin, cout, t, h, w

                new_w = ms.Parameter(new_w, name=key_3d)

                new_state_dict[key_3d] = new_w
            elif "attn_1" in key_2d:
                new_val = vae2d_sd[key_2d].expand_dims(axis=2)
                new_param = ms.Parameter(new_val, name=key_3d)
                new_state_dict[key_3d] = new_param
            else:
                raise NotImplementedError(f"Key {key_3d} ({key_2d}) not implemented")

            m, u = ms.load_param_into_net(self, new_state_dict)
            if len(m) > 0:
                logger.info("net param not loaded: ", m)
            if len(u) > 0:
                logger.info("checkpoint param not loaded: ", u)

    def init_from_ckpt(
        self,
        path,
        ignore_keys=list(),
        remove_prefix=["first_stage_model.", "autoencoder."],
    ):
        # TODO: support auto download pretrained checkpoints
        sd = ms.load_checkpoint(path)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    logger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        ms.load_param_into_net(self, sd, strict_load=False)
        logger.info(f"Restored from {path}")

    def encode(self, x):
        self.set_train(False)
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        quantized, _ = self.quantizer(z_e)
        return quantized

    def decode(self, x):
        self.set_train(False)
        return self.decoder(x)

    def decode_from_indices(self, ids: np.ndarray):
        features = self.quantizer.decode_ids(ids)
        reconstructed_video = self.decode(features)
        return reconstructed_video

    # def decode_from_ids(self, ids):

    #     if self.lookup_free_quantization:
    #         ids, ps = pack([ids], 'b *')
    #         fmap = self.quantizer.indices_to_codes(ids)
    #         fmap, = unpack(fmap, ps, 'b * c')
    #     else:
    #         codes = self.codebook[ids]
    #         fmap = self.quantizer.project_out(codes)

    #     fmap = rearrange(fmap, 'b h w c -> b c h w')
    #     return self.decode(fmap)

    def decode_stage1(self, ids: np.ndarray):
        features = self.quantizer.decode_ids(ids)
        pre_activation_embeddings = self.decoder(features, mode="stage1")
        return pre_activation_embeddings

    def decode_stage2(self, embeddings: np.ndarray):
        reconstructed_video = self.decoder(embeddings, mode="stage2")
        return reconstructed_video

    def encode_to_indices(self, inputs: np.ndarray):
        _, result_dict = self.encode(inputs)
        ids = result_dict["encoding_indices"]
        return ids

    def get_encoded_fmap_size(self, video_size):
        return video_size // (2**self.config.vqvae.num_enc_res_blocks)

    def construct(self, x):

        encode_first_frame_separately = (
            self.separate_first_frame_encoding and self.video_contains_first_frame
        )

        # whether to pad video or not

        if self.video_contains_first_frame:
            video_len = x.shape[2]
            x = pad_at_dim(x, (self.time_padding, 0), value=0.0, dim=2)
            # video_packed_shape = [self.time_padding, 1, video_len - 1]

        # initial conv
        # taking into account whether to encode first frame separately

        if encode_first_frame_separately:
            pad = x[:, :, : self.time_padding, :, :]
            first_frame = x[:, :, self.time_padding, :, :]
            x = x[:, :, -(video_len - 1) :, :, :]
            # pad, first_frame, video = unpack(x, video_packed_shape, 'b c * h w')
            first_frame = self.conv_in_first_frame(first_frame)
        else:
            first_frame = x[:, :, self.time_padding, :, :]

        x = self.conv_in(x)

        if encode_first_frame_separately:
            first_frame = first_frame.unsqueeze(2)
            x = ops.cat([first_frame, x], axis=2)
            x = pad_at_dim(x, (self.time_padding, 0), dim=2)

        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, aux_loss = self.quantizer(z_e)

        # decode
        x_hat = self.decoder(z_q)

        decode_first_frame_separately = (
            self.separate_first_frame_encoding and self.video_contains_first_frame
        )

        if decode_first_frame_separately:
            left_pad, xff, x_hat = (
                x_hat[:, :, : self.time_padding],
                x_hat[:, :, self.time_padding],
                x_hat[:, :, (self.time_padding + 1) :],
            )

            out = self.conv_out(x_hat)
            outff = self.conv_out_first_frame(xff)

            video = ops.cat([outff.unsqueeze(2), out], axis=2)

        else:
            video = self.conv_out(x_hat)

            # if video were padded, remove padding

            if self.video_contains_first_frame:
                video = video[:, :, self.time_padding :]

        return z_e, z_q, video, aux_loss
