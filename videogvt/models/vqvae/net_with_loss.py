import mindspore as ms
from mindspore import nn, ops


from .lpips import LPIPS


@ms.jit
def _rearrange_in(x):
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4)
    x = ops.reshape(x, (b * t, c, h, w))

    return x


class GeneratorWithLoss(nn.Cell):
    def __init__(
        self,
        vqvae,
        disc_start=50001,
        disc_weight=0.1,
        disc_factor=1.0,
        perceptual_weight=1.0,
        recons_weight=5.0,
        discriminator=None,
        dtype=ms.float32,
        **kwargs,
    ):
        super().__init__()

        # build perceptual models for loss compute
        self.vqvae = vqvae
        # TODO: set dtype for LPIPS ?
        self.perceptual_loss = LPIPS()  # freeze params inside
        self.perceptual_loss.to_float(dtype)

        self.l1 = nn.L1Loss(reduction="none")

        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.disc_factor = disc_factor
        self.recons_weight = recons_weight
        self.perceptual_weight = perceptual_weight

        self.discriminator = discriminator
        if (self.discriminator is not None) and (self.disc_factor > 0.0):
            self.has_disc = True
        else:
            self.has_disc = False

        self.dtype = dtype

    def loss_function(
        self,
        x,
        recons,
        aux_loss,
        cond=None,
    ):
        bs = x.shape[0]
        x_reshape = _rearrange_in(x)
        recons_reshape = _rearrange_in(recons)

        # 2.1 entropy loss and commitment loss
        loss = aux_loss

        # 2.2 reconstruction loss in pixels
        rec_loss = self.l1(x_reshape, recons_reshape) * self.recons_weight

        # 2.3 perceptual loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(x_reshape, recons_reshape)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        loss += rec_loss.mean()

        # 2.4 discriminator loss if enabled
        g_loss = ms.Tensor(0.0, dtype=self.dtype)
        if self.has_disc:
            # calc gan loss
            if cond is None:
                logits_fake = self.discriminator(recons)
            else:
                logits_fake = self.discriminator(ops.concat((recons, cond), dim=1))
            g_loss = -ops.mean(logits_fake)

        # TODO: do adaptive weighting based on grad
        # d_weight = self.calculate_adaptive_weight(mean_nll_loss, g_loss, last_layer=last_layer)
        d_weight = self.disc_weight
        loss += d_weight * self.disc_factor * g_loss
        # print(f"nll_loss: {mean_weighted_nll_loss.asnumpy():.4f}, kl_loss: {kl_loss.asnumpy():.4f}")

        """
        split = "train"
        log = {"{}/total_loss".format(split): loss.asnumpy().mean(),
           "{}/logvar".format(split): self.logvar.value().asnumpy(),
           "{}/kl_loss".format(split): kl_loss.asnumpy().mean(),
           "{}/nll_loss".format(split): nll_loss.asnumpy().mean(),
           "{}/rec_loss".format(split): rec_loss.asnumpy().mean(),
           # "{}/d_weight".format(split): d_weight.detach(),
           # "{}/disc_factor".format(split): torch.tensor(disc_factor),
           # "{}/g_loss".format(split): g_loss.detach().mean(),
           }
        for k in log:
            print(k.split("/")[1], log[k])
        """
        # TODO: return more losses

        return loss

    # in graph mode, construct code will run in graph. TODO: in pynative mode, need to add ms.jit decorator
    def construct(
        self,
        x: ms.Tensor,
        cond=None,
    ):
        """
        x: input images or videos, images: (b c h w), videos: (b c t h w)
        global_step: global training step
        """

        # 1. AE forward, get aux_loss (entropy + commitment loss) and recons
        _, _, recons, aux_loss = self.vqvae(x)

        # For videos, treat them as independent frame images
        # TODO: regularize on temporal consistency
        # if x.ndim >= 5:
        # x: b c t h w -> (b*t c h w), shape for image perceptual loss

        # 2. compuate loss
        loss = self.loss_function(x, recons, aux_loss, cond)

        return loss.astype(self.dtype)


class DiscriminatorWithLoss(nn.Cell):
    """
    Training logic:
        For training step i, input data x:
            1. AE generator takes input x, feedforward to get posterior/latent and reconstructed data, and compute ae loss
            2. AE optimizer updates AE trainable params
            3. D takes the same input x, feed x to AE again **again** to get
                the new posterior and reconstructions (since AE params has updated), feed x and recons to D, and compute D loss
            4. D optimizer updates D trainable params
            --> Go to next training step
        Ref: sd-vae training
    """

    def __init__(
        self,
        vqvae,
        discriminator,
        disc_start=50001,
        disc_factor=1.0,
        disc_loss="hinge",
    ):
        super().__init__()
        self.vqvae = vqvae
        self.discriminator = discriminator
        self.disc_start = disc_start
        self.disc_factor = disc_factor

        assert disc_loss in ["hinge", "vanilla"]
        if disc_loss == "hinge":
            self.disc_loss = self.hinge_loss
        else:
            self.softplus = ops.Softplus()
            self.disc_loss = self.vanilla_d_loss

    def hinge_loss(self, logits_real, logits_fake):
        loss_real = ops.mean(ops.relu(1.0 - logits_real))
        loss_fake = ops.mean(ops.relu(1.0 + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def vanilla_d_loss(self, logits_real, logits_fake):
        d_loss = 0.5 * (
            ops.mean(self.softplus(-logits_real)) + ops.mean(self.softplus(logits_fake))
        )
        return d_loss

    def construct(self, x: ms.Tensor):
        """
        Second pass
        Args:
            x: input image/video, (bs c h w)
            weights: sample weights
        """

        # 1. AE forward, get posterior (mean, logvar) and recons
        _, _, recons, _ = self.vqvae(x)

        # 2. Disc forward to get class prediction on real input and reconstrucions

        logits_real = self.discriminator(x)
        logits_fake = self.discriminator(recons)

        # logits_real = self.discriminator(ops.concat((x, cond), dim=1))
        # logits_fake = self.discriminator(ops.concat((recons, cond), dim=1))

        d_loss = self.disc_factor * self.disc_loss(logits_real, logits_fake)

        return d_loss
