# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_base import (
    TransformerModelBase,
)
from fairseq.models.transformer import TransformerEncoder
from .s2t_ms_transformer import TransformerDecoderScriptable
from fairseq.modules import LayerNorm
from torch import nn

def mix_input(audio, source, align_pad, align_lengths, prob):
    # token replace
    audio = audio.transpose(0, 1)
    source = source.transpose(0, 1)
    mixseq = []
    bsz, _, fdim = audio.shape
    if not isinstance(prob, list):
        prob = [prob for i in range(bsz)]
    for i in range(bsz):
        word_length = align_lengths[i].item()
        if word_length != 0:
            word_prob = torch.rand(word_length)
            word_sample = word_prob < prob[i]
            seq = torch.zeros(0, fdim).cuda().type_as(audio)
            for j in range(word_length):
                if word_sample[j]:
                    st, en = align_pad[i, j, 0:2]
                    audio_seq = audio[i, st:en, :]
                    seq = torch.cat((seq, audio_seq), dim=0)
                else:
                    st, en = align_pad[i, j, 2:4]
                    text_seq = source[i, st:en, :]
                    seq = torch.cat((seq, text_seq), dim=0)
        else:
            seq = source[i]
        mixseq.append(seq)
    mixseq_length = torch.LongTensor([seq.size(0) for seq in mixseq]).cuda()
    max_len = torch.max(mixseq_length).item()
    mixseq_pad = torch.zeros(bsz, max_len, fdim).cuda().type_as(audio)
    for i, seq in enumerate(mixseq):
        mixseq_pad[i, :seq.size(0)] = seq
    mixseq_encoder_padding_mask = lengths_to_padding_mask(mixseq_length)
    mixseq_pad = mixseq_pad.transpose(0, 1)
    return mixseq_pad, mixseq_encoder_padding_mask

@register_model("ms_srk_transformer")
class MSSrkTransformerModel(TransformerModelBase):
    """
    This is the legacy implementation of the transformer model that
    uses argparse for configuration.
    """

    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )
        parser.add_argument(
            "--src-word-subsample-layer",
            type=int,
            metavar="N",
            help="~",
        )
        parser.add_argument(
            "--tgt-word-subsample-layer",
            type=int,
            metavar="N",
            help="~",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )

        args.src_word_dict_size = len(task.source_word_dictionary) if task.source_word_dictionary is not None else None
        args.tgt_word_dict_size = len(task.target_word_dictionary) if task.target_word_dictionary is not None else None

        cfg = TransformerConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            TransformerConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return MSTransformerEncoder(
            TransformerConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderScriptable(
            TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )

    def get_ctc_target(self, sample: Optional[Dict[str, Tensor]], scale: str):
        tgt_idx = "target" + scale
        len_idx = "target_lengths" + scale
        return sample[tgt_idx], sample[len_idx]

    def get_ctc_output(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        sample: Optional[Dict[str, Tensor]],
        scale: str
    ):
        output_idx = "encoder_out" + scale
        ctc_proj_ = "ctc_proj" + scale
        mask_idx = "encoder_padding_mask" + scale
        encoder_out = net_output[1]["encoder_out"][output_idx][0]
        logits = self.encoder.ctc_proj[ctc_proj_](encoder_out)  # T x B x C
        out = utils.log_softmax(logits.float(), dim=-1)
        padding_mask = net_output[1]["encoder_out"][mask_idx]
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens


class MSTransformerEncoder(TransformerEncoder):
    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        super().__init__(cfg, dictionary, embed_tokens, return_fc=return_fc)

        self.layer_norm_src_word = LayerNorm(cfg.encoder_embed_dim)
        self.layer_norm_tgt_word = LayerNorm(cfg.encoder_embed_dim)

        self.src_word_subsample_layer = cfg.src_word_subsample_layer
        self.tgt_word_subsample_layer = cfg.tgt_word_subsample_layer
        #self.contrastive_weight_phone = cfg.contrastive_weight_phone
        #self.contrastive_weight_word = cfg.contrastive_weight_word

        self.mixup_rate_phone = getattr(cfg, "mixup_rate_phone", 0)
        self.mixup_rate_word = getattr(cfg, "mixup_rate_word", 0)
        self.contrastive_weight_word = getattr(cfg, "contrastive_weight_word", 0)
        self.contrastive_weight_phone = getattr(cfg, "contrastive_weight_phone", 0)

        self.ctc_proj = None
        if getattr(cfg, "ctc_weight", 0.0) > 0.0:
            self.ctc_src_word = nn.Linear(cfg.encoder_embed_dim, cfg.src_word_dict_size) \
                if cfg.src_word_dict_size is not None else None
            self.ctc_tgt_word = nn.Linear(cfg.encoder_embed_dim, cfg.tgt_word_dict_size) \
                if cfg.tgt_word_dict_size is not None else None
            self.ctc_proj = {"ctc_proj_src_word": self.ctc_src_word, "ctc_proj_tgt_word": self.ctc_tgt_word}

    def get_mixed_input(self, audio, source, align_pad, align_lengths, prob):
        mix_output = mix_input(audio, source, align_pad, align_lengths, prob)
        mixseq, mixseq_encoder_padding_mask = mix_output
        positions = self.embed_positions(mixseq_encoder_padding_mask).transpose(0, 1)
        mixseq += positions
        if self.layernorm_embedding is not None:
            mixseq = self.layernorm_embedding(mixseq)
        mixseq = self.dropout_module(mixseq)
        return mixseq, mixseq_encoder_padding_mask

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        spch_enc_state = None,
        align_pad = None,
        align_lengths = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, spch_enc_state, align_pad, align_lengths, token_embeddings
        )

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        spch_enc_state = None,
        align_pad = None,
        align_lengths = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # nested tensor and BT enable
        layer = self.layers[0]
        BT_flag = False
        NT_flag = False
        # torch version check, BT>=1.12.0 and NT>=1.13.0.dev20220613
        # internal format is '1.13.0a0+fb'
        # external format is '1.13.0.dev20220613'(cpu&gpu) for nightly or "1.11.0"(cpu) or '1.11.0+cu102'(gpu) for stable
        BT_version = False
        NT_version = False
        if "fb" in torch.__version__:
            BT_version = True
            NT_version = True
        else:
            if "+" in torch.__version__:
                torch_version = torch.__version__.split("+")[0]
            else:
                torch_version = torch.__version__

            torch_version = torch_version.split(".")
            int_version = (
                int(torch_version[0]) * 1000
                + int(torch_version[1]) * 10
                + int(torch_version[2])
            )
            if len(torch_version) == 3:
                if int_version >= 1120:
                    BT_version = True
                if int_version >= 1131:
                    NT_version = True
            elif len(torch_version) == 4:
                if int_version >= 1130:
                    BT_version = True
                # Consider _nested_tensor_from_mask_left_aligned is landed after "20220613"
                if int_version >= 1131 or (
                        int_version == 1130 and torch_version[3][3:] >= "20220613"
                ):
                    NT_version = True

        if (
            BT_version
            and x.dim() == 3
            and layer.load_to_BT
            and not layer.return_fc
            and layer.can_use_fastpath
            and not layer.training
            and not layer.ever_training
            and not layer.cfg_checkpoint_activations
        ):
            # Batch first can not be justified but needs user to make sure
            x = x.transpose(0, 1)
            # Check mask conditions for nested tensor
            if NT_version:
                if (
                        encoder_padding_mask is not None
                        and torch._nested_tensor_from_mask_left_aligned(
                    x, encoder_padding_mask.logical_not()
                )
                ):
                    if not torch.is_grad_enabled() or not x.requires_grad:
                        x = torch._nested_tensor_from_mask(
                            x, encoder_padding_mask.logical_not()
                        )
                        NT_flag = True
            BT_flag = True

        # encoder layers
        if NT_flag:
            processing_mask = None
        else:
            processing_mask = encoder_padding_mask
        # encoder_padding_mask_out = processing_mask if has_pads else None
        encoder_padding_mask_out = processing_mask

        src_word_state = None
        tgt_word_state = None
        text_contrastive_phone_state = None
        text_contrastive_word_state = None
        encoder_padding_mask_src_word = None
        encoder_padding_mask_tgt_word = None
        text_encoder_padding_mask_contrastive_phone = None
        text_encoder_padding_mask_contrastive_word = None

        for i, layer in enumerate(self.layers):

            # phone ctr
            if self.contrastive_weight_phone > 0 and i == 0:
                text_contrastive_phone_state = x
                text_encoder_padding_mask_contrastive_phone = encoder_padding_mask_out

            # mixup phone
            if self.mixup_rate_phone > 0 and spch_enc_state is not None and i == 0 and align_lengths is not None and align_pad is not None:
                """
                要加position embedding
                """
                spch_mixup_state_phone = spch_enc_state['spch_mixup_state_phone'][0]
                spch_mixup_padding_mask_phone = spch_enc_state['spch_mixup_padding_mask_phone'][0]

                x, encoder_padding_mask_out = self.get_mixed_input(spch_mixup_state_phone, x,
                                                                       align_pad, align_lengths, self.mixup_rate_phone)


            lr = layer(x, encoder_padding_mask=encoder_padding_mask_out)

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            # word ctc
            if i + 1 == self.src_word_subsample_layer and self.ctc_proj["ctc_proj_src_word"] is not None:
                # x = self.layer_norm_src_word(x)
                """
                如果加shrink，别忘了改mask
                """
                src_word_state = x
                encoder_padding_mask_src_word = encoder_padding_mask_out

                # # mixup word
                # if self.mixup_rate_word > 0 and spch_enc_state is not None:
                #     mixup_state_word = x
                #     mixup_padding_mask_word = encoder_padding_mask_out

            # word ctr
            if i + 1 == self.src_word_subsample_layer and self.contrastive_weight_word > 0:
                text_contrastive_word_state = x
                text_encoder_padding_mask_contrastive_word = encoder_padding_mask_out

            # tgt word ctc
            if i + 1 == self.tgt_word_subsample_layer and self.ctc_proj["ctc_proj_tgt_word"] is not None:
                # x = self.layer_norm_tgt_word(x)
                tgt_word_state = x
                encoder_padding_mask_tgt_word = encoder_padding_mask_out

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        # change back to non-nested and Batch second
        if NT_flag:
            x = x.to_padded_tensor(0.0)

        if NT_flag or BT_flag:
            x = x.transpose(0, 1)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_out_src_word": [src_word_state] if src_word_state is not None else [],
            "encoder_out_tgt_word": [tgt_word_state] if tgt_word_state is not None else [],
            "text_contrastive_phone_state": [text_contrastive_phone_state] if text_contrastive_phone_state is not None else [],
            "text_contrastive_word_state": [text_contrastive_word_state] if text_contrastive_word_state is not None else [],
            "encoder_padding_mask_src_word": [encoder_padding_mask_src_word]
            if encoder_padding_mask_src_word is not None and encoder_padding_mask_src_word.any()
            else [],  # B x T
            "encoder_padding_mask_tgt_word": [encoder_padding_mask_tgt_word]
            if encoder_padding_mask_tgt_word is not None and encoder_padding_mask_tgt_word.any()
            else [],  # B x T
            "text_encoder_padding_mask_contrastive_phone": [text_encoder_padding_mask_contrastive_phone]
            if text_encoder_padding_mask_contrastive_phone is not None
            else [],  # B x T
            "text_encoder_padding_mask_contrastive_word": [text_encoder_padding_mask_contrastive_word]
            if text_encoder_padding_mask_contrastive_word is not None
            else [],  # B x T
            "encoder_padding_mask": [encoder_padding_mask_out],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }


# architectures
@register_model_architecture("ms_srk_transformer", "ms_srk_transformer")
def base_architecture(args):
    args.src_word_subsample_layer = getattr(args, "src_word_subsample_layer", 2)
    args.tgt_word_subsample_layer = getattr(args, "tgt_word_subsample_layer", 6)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
