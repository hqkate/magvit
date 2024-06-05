import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Normal

from videogvt.models.vqvae.model_utils import CausalConv3d, GroupNormExtend


# Copied from https://github.com/wilson1yan/VideoGPT
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


# Copied from https://github.com/wilson1yan/VideoGPT
def shift_dim(x, src_dim=-1, dest_dim=-1):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim
    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims
    dims = list(range(n_dims))
    # del dims[src_dim]
    dims = dims[:src_dim] + dims[src_dim + 1 :]
    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    return x


# Copied from https://github.com/wilson1yan/VideoGPT
class AxialAttention(nn.Cell):
    def __init__(self, n_dim, axial_dim, causal=False, dtype=ms.float32):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2  # account for batch, head, dim
        self.causal = causal
        self.axial_dim = axial_dim
        self.dtype = dtype

    def scaled_dot_product_attention(self, q, k, v, mask=None, attn_dropout=0.0):
        # Performs scaled dot-product attention over the second to last dimension dn

        # (b, n_head, d1, ..., dn, d)
        attn = ops.matmul(q, ops.swapaxes(k, -1, -2))
        attn = attn / np.sqrt(q.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(mask == 0, ms.Tensor(float("-inf"), attn.dtype))
        attn_float = ops.softmax(attn, axis=-1)
        attn = attn_float.type_as(attn)  # b x n_head x d1 x ... x dn x d
        attn = ops.dropout(attn, p=attn_dropout, training=self.training)

        a = ops.matmul(attn, v)  # b x n_head x d1 x ... x dn x d

        return a

    def construct(self, q, k, v, decode_step, decode_idx):
        # batch, head, frame, height, width, dim
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)

        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        if self.causal:
            mask = (
                ops.tril(ops.ones((q.shape[-2], q.shape[-2]), dtype=self.dtype))
                if self.causal
                else None
            )
            if decode_step is not None and mask is not None:
                mask = mask[[decode_step]]
        else:
            mask = None

        out = self.scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


# Copied from https://github.com/wilson1yan/VideoGPT
class MultiHeadAttention(nn.Cell):
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer, causal, attn_kwargs, dtype=ms.float32):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = nn.Dense(
            dim_q,
            n_head * self.d_k,
            has_bias=False,
            weight_init=Normal(sigma=1.0 / np.sqrt(dim_q)),
            dtype=dtype
        )  # q
        # self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Dense(
            dim_kv,
            n_head * self.d_k,
            has_bias=False,
            weight_init=Normal(sigma=1.0 / np.sqrt(dim_kv)),
            dtype=dtype
        )  # k
        # self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Dense(
            dim_kv,
            n_head * self.d_v,
            has_bias=False,
            weight_init=Normal(sigma=1.0 / np.sqrt(dim_kv)),
            dtype=dtype
        )  # v
        # self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Dense(
            n_head * self.d_v,
            dim_q,
            has_bias=True,
            weight_init=Normal(sigma=1.0 / np.sqrt(dim_q * n_layer)),
            dtype=dtype
        )  # c
        # self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        self.attn = AxialAttention(len(shape), causal=causal, dtype=dtype, **attn_kwargs)

        self.cache = None

    def construct(self, q, k, v, decode_step=None, decode_idx=None):
        """Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None

        Returns
            The output after performing attention
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (q.shape[0], n_head, *self.shape, self.d_k)
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(
                        k=ops.zeros(k_shape, dtype=k.dtype),
                        v=ops.zeros(v_shape, dtype=v.dtype),
                    )
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.copy(), v=v.copy())
            if self.causal:
                idx = (
                    slice(None, None),
                    slice(None, None),
                    *[slice(i, i + 1) for i in decode_idx],
                )
                self.cache["k"][idx] = k
                self.cache["v"][idx] = v
            k, v = self.cache["k"], self.cache["v"]

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a)  # (b x seq_len x embd_dim)

        return a


# Modified from https://github.com/wilson1yan/VideoGPT
class AxialBlock(nn.Cell):
    def __init__(self, n_hiddens, n_head, dtype=ms.float32):
        super().__init__()
        kwargs = dict(
            shape=(0,) * 3,
            dim_q=n_hiddens,
            dim_kv=n_hiddens,
            n_head=n_head,
            n_layer=1,
            causal=False,
        )
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2), **kwargs, dtype=dtype)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3), **kwargs, dtype=dtype)
        kwargs["causal"] = True
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4), **kwargs, dtype=dtype)

    def construct(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


# Copied from https://github.com/wilson1yan/VideoGPT
class AttentionResidualBlock(nn.Cell):
    def __init__(self, n_hiddens, n_heads: int = 2, dtype=ms.float32):
        super().__init__()
        self.block = nn.SequentialCell(
            GroupNormExtend(
                num_groups=32, num_channels=n_hiddens, dtype=dtype
            ),  # nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            CausalConv3d(n_hiddens, n_hiddens // 2, 3, has_bias=False),
            GroupNormExtend(
                num_groups=16, num_channels=n_hiddens // 2, dtype=dtype
            ),  # nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            CausalConv3d(n_hiddens // 2, n_hiddens, 1, has_bias=False),
            GroupNormExtend(
                num_groups=32, num_channels=n_hiddens, dtype=dtype
            ),  # nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, n_heads, dtype=dtype),
        ).to_float(dtype)

    def construct(self, x):
        return x + self.block(x)


if __name__ == "__main__":
    from mindspore import context

    ascend_config = {"precision_mode": "allow_mix_precision_bf16"}
    context.set_context(
        mode=1,
        device_target="Ascend",
        device_id=1,
        ascend_config=ascend_config,
    )

    # random data
    x = np.random.random_sample((2, 8, 16, 32, 32))
    x = ms.Tensor(x, ms.float16)

    # test AttentionResidualBlock
    res = AttentionResidualBlock(8, 4, dtype=ms.float16).to_float(ms.float16)
    res_out = res(x)
    print("AttentionResidualBlock out shape:", res_out.shape)
