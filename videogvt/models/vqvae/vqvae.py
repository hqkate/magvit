import mindspore as ms
import numpy as np
from mindspore import nn, ops

from videogvt.models.vqvae.encoder import Encoder, Encoder3D
from videogvt.models.vqvae.vector_quantizer import VQ
from videogvt.models.vqvae.lookup_free_quantization import LFQ
from videogvt.models.vqvae.decoder import Decoder, Decoder3D


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
        is_training=False,
    ):
        super(VQVAE3D, self).__init__()
        self.config = config

        # encode image into continuous latent space
        self.encoder = Encoder3D(config)
        # decode the discrete latent representation
        self.decoder = Decoder3D(config)

        embedding_dim = config.vqvae.embedding_dim
        h_dim = config.vqvae.filters
        m_dim = config.vqvae.middle_channles
        beta = config.vqvae.commitment_cost
        self.codebook_size = config.vqvae.codebook_size
        self.pre_quantization_conv = nn.Conv3d(m_dim, m_dim, kernel_size=1, stride=1, dtype=ms.float16)
        # pass continuous latent vector through discretization bottleneck
        if lookup_free_quantization:
            self.quantizer = LFQ(
                dim=m_dim,
                codebook_size=self.codebook_size,
                return_loss_breakdown=False,
                is_training=is_training,
                # **lfq_kwargs
            )
        else:
            self.quantizer = VQ(self.codebook_size, embedding_dim, beta)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(self.codebook_size)}
        else:
            self.img_to_embedding_map = None

    def construct(self, x):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        z_e = z_e.astype(ms.float32)
        z_q, aux_loss = self.quantizer(z_e)
        z_q = z_q.astype(ms.float16)
        x_hat = self.decoder(z_q)

        return z_e, z_q, x_hat, aux_loss
