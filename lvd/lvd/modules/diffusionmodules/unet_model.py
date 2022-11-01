import torch
import torch.nn as nn
from .util import timestep_embedding
from lvd.modules.transformers import SpatialTransformer, AttentionBlock
from lvd.modules.diffusionmodules.util import (checkpoint, conv_nd, linear, avg_pool_nd, zero_module, normalization,
                                               timestep_embedding, )
import torch.nn.functional as F
from abc import abstractmethod


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False,
                 dims=2, use_checkpoint=False, up=False, down=False, ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(),
                                       conv_nd(dims, channels, self.out_channels, 3, padding=1) if dims != 3 else
                                       nn.Conv3d(channels, self.out_channels, kernel_size=(1, 3, 3), stride=1,
                                                 padding=(0, 1, 1)), )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels,
                                                          2 * self.out_channels if use_scale_shift_norm else self.out_channels, ), )
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)) if dims != 3 else nn.Conv3d(
            self.out_channels, self.out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)), )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNetModel(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions,
                 dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None,
                 use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, use_scale_shift_norm=False,
                 resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False,
                 # custom transformer support
                 transformer_depth=1,  # custom transformer support
                 context_dim=None,  # custom transformer support
                 n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
                 legacy=True, ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'context dimension cannot be none for spatial transformer'
        if context_dim is not None:
            assert use_spatial_transformer, "using spatial transformer for cross-attention conditioning"

        assert num_heads != -1 or num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(),
                                        nn.Linear(time_embed_dim, time_embed_dim))
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])

        ################################# the input blocks #############################
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims,
                                   use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels  # multi-head attention
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, dims=dims,
                                                     context_dim=context_dim) if use_spatial_transformer else AttentionBlock(
                        ch, use_checkpoint=use_checkpoint, num_head_channels=dim_head, num_heads=num_heads,
                        use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                             use_scale_shift_norm=use_scale_shift_norm, down=True, ) if resblock_updown else Downsample(
                        ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        ################################# the middle blocks #############################
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm, ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head,
                           use_new_attention_order=use_new_attention_order, ) if not use_spatial_transformer else
            SpatialTransformer(ch, num_heads, dim_head, dims=dims, depth=transformer_depth, context_dim=context_dim),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm, ), )
        self._feature_size += ch

        ################################# the out blocks #############################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims,
                                   use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, )]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, dims=dims,
                                                     context_dim=context_dim) if use_spatial_transformer else
                                  AttentionBlock(ch, use_checkpoint=use_checkpoint, num_head_channels=dim_head,
                                                 num_heads=num_heads, use_new_attention_order=use_new_attention_order))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, up=True,
                                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, )
                                  if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(normalization(ch), nn.SiLU(),
                                 zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)), )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(normalization(ch), conv_nd(dims, model_channels, n_embed, 1), )

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class UNet3D(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions,
                 dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=3, num_classes=None,
                 use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, use_scale_shift_norm=False,
                 resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False,
                 transformer_depth=1, context_dim=None, n_embed=None, legacy=True, ):
        super().__init__()
        assert dims==3, 'only support 3 dimensions'
        self.num_heads = num_heads
        self.model_channels = model_channels
        self.attention_resolutions = attention_resolutions
        self.num_head_channels = num_head_channels
        self.use_spatial_transformer = use_spatial_transformer
        self.legacy = legacy
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim)
        )
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv3d(in_channels, model_channels, kernel_size=(1, 3, 3), stride=1,
                                               padding=(0, 1, 1)))])

        ################################# the input blocks #############################
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims,
                                   use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]

                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = self.update_head_info(ch)

                    layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=transformer_depth, dims=dims,
                                                     context_dim=context_dim) if use_spatial_transformer else AttentionBlock(
                        ch, use_checkpoint=use_checkpoint, num_head_channels=dim_head, num_heads=self.num_heads,
                        use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                             use_scale_shift_norm=use_scale_shift_norm, down=True, ) if resblock_updown else Downsample(
                        ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        ################################# the middle blocks #############################
        dim_head = self.update_head_info(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm, ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=self.num_heads, num_head_channels=dim_head,
                           use_new_attention_order=use_new_attention_order, ) if not use_spatial_transformer else
            SpatialTransformer(ch, self.num_heads, dim_head, dims=dims, depth=transformer_depth,
                               context_dim=context_dim),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm, ), )
        self._feature_size += ch

        ################################# the output blocks ################################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims,
                                   use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, )]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    dim_head = self.update_head_info(ch)
                    layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=transformer_depth, dims=dims,
                                                     context_dim=context_dim) if use_spatial_transformer else
                                  AttentionBlock(ch, use_checkpoint=use_checkpoint, num_head_channels=dim_head,
                                                 num_heads=self.num_heads,
                                                 use_new_attention_order=use_new_attention_order))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, up=True,
                                           use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, )
                                  if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(normalization(ch), nn.SiLU(),
                                 zero_module(nn.Conv3d(in_channels, model_channels, kernel_size=(1, 3, 3), stride=1,
                                                       padding=(0, 1, 1))), )

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(normalization(ch), conv_nd(dims, model_channels, n_embed, 1), )

    def update_head_info(self, ch):
        if self.num_head_channels == -1:
            dim_head = ch // self.num_heads
        else:
            self.num_heads = ch // self.num_head_channels  # multi-head attention
            dim_head = self.num_head_channels
        if self.legacy:
            dim_head = ch // self.num_heads if self.use_spatial_transformer else self.num_head_channels

        return dim_head

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
