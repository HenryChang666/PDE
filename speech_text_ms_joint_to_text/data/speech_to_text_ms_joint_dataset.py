# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import torch

from fairseq.data import ConcatDataset, Dictionary, ResamplingDataset
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
)

logger = logging.getLogger(__name__)


class S2TMSJointDataConfig(S2TDataConfig):
    """Wrapper class for data config YAML"""

    @property
    def src_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("src_vocab_filename", "src_dict.txt")

    @property
    def src_word_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("src_word_vocab_filename", "spm_ende_dict.txt")

    @property
    def src_phone_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("src_phone_vocab_filename", "src_phone_dict.txt")

    @property
    def src_char_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("src_char_vocab_filename", "dict_c.en.txt")

    @property
    def src_pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("src_pre_tokenizer", {"tokenizer": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def src_bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply on source text after pre-tokenization.
        Returning a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("src_bpe_tokenizer", {"bpe": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def src_word_pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("src_word_pre_tokenizer", {"tokenizer": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def src_word_bpe_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("src_word_bpe_tokenizer", {"bpe": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def prepend_tgt_lang_tag_no_change(self) -> bool:
        """Prepend target lang ID token as the prev_output_tokens BOS (e.g. for
        to-many multilingual setting). No change needed during inference.
        This option is deprecated and replaced by prepend_tgt_lang_tag_as_bos.
        """
        value = self.config.get("prepend_tgt_lang_tag_no_change", None)
        if value is None:
            return self.config.get("prepend_tgt_lang_tag_as_bos", False)
        return value

    @property
    def sampling_text_alpha(self):
        """Hyper-parameter alpha = 1/T for temperature-based resampling. (text
        input only) (alpha = 1 for no resampling)"""
        return self.config.get("sampling_text_alpha", 1.0)


class SpeechToTextMSJointDatasetItem(NamedTuple):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    src_txt_word_tokens: Optional[torch.Tensor] = None
    src_txt_phone_tokens: Optional[torch.Tensor] = None
    src_txt_char_tokens: Optional[torch.Tensor] = None
    src_txt_tokens: Optional[torch.Tensor] = None
    spch_txt_align: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None
    src_lang_tag: Optional[int] = None
    tgt_alignment: Optional[torch.Tensor] = None


# use_src_lang_id:
#   0: don't use src_lang_id
#   1: attach src_lang_id to the src_txt_tokens as eos
class SpeechToTextMSJointDataset(SpeechToTextDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2TMSJointDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        src_word_texts: Optional[List[str]] = None,
        src_phone_texts: Optional[List[str]] = None,
        src_char_texts: Optional[List[str]] = None,
        speech_text_aligns = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        src_dict: Optional[Dictionary] = None,
        src_word_dict: Optional[Dictionary] = None,
        src_phone_dict: Optional[Dictionary] = None,
        src_char_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        src_pre_tokenizer=None,
        src_bpe_tokenizer=None,
        src_word_pre_tokenizer=None,
        src_word_bpe_tokenizer=None,
        append_eos: Optional[bool] = True,
        alignment: Optional[List[str]] = None,
        use_src_lang_id: Optional[int] = 0,
    ):
        super().__init__(
            split,
            is_train_split,
            cfg,
            audio_paths,
            n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            speakers=speakers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            append_eos=append_eos,
        )

        assert src_word_texts is None or len(src_word_texts) == self.n_samples
        assert src_phone_texts is None or len(src_phone_texts) == self.n_samples
        assert src_char_texts is None or len(src_char_texts) == self.n_samples
        # assert (src_word_dict is None and src_word_texts is None) or (
        #         src_word_dict is not None and src_word_texts is not None
        # )
        # assert (src_phone_dict is None and src_phone_texts is None) or (
        #         src_phone_dict is not None and src_phone_texts is not None
        # )
        # assert (src_char_dict is None and src_char_texts is None) or (
        #         src_char_dict is not None and src_char_texts is not None
        # )
        self.src_word_texts, self.src_phone_texts, self.src_char_texts = src_word_texts, src_phone_texts, src_char_texts
        self.speech_text_aligns = speech_text_aligns
        self.src_dict = src_dict
        self.src_word_dict, self.src_phone_dict, self.src_char_dict = src_word_dict, src_phone_dict, src_char_dict
        self.src_pre_tokenizer = src_pre_tokenizer
        self.src_bpe_tokenizer = src_bpe_tokenizer
        self.src_word_pre_tokenizer = src_word_pre_tokenizer
        self.src_word_bpe_tokenizer = src_word_bpe_tokenizer
        self.alignment = None
        self.use_src_lang_id = use_src_lang_id
        if alignment is not None:
            self.alignment = [
                [float(s) for s in sample.split()] for sample in alignment
            ]

    def get_tokenized_src_text(self, index: int):
        text = self.tokenize(self.src_pre_tokenizer, self.src_texts[index])
        text = self.tokenize(self.src_bpe_tokenizer, text)
        return text

    def get_tokenized_src_word_text(self, index: int):
        src_word_text = self.tokenize(self.src_word_pre_tokenizer, self.src_word_texts[index])
        src_word_text = self.tokenize(self.src_word_bpe_tokenizer, src_word_text)
        return src_word_text

    def get_tokenized_src_phone_text(self, index: int):
        phone_text = self.tokenize(None, self.src_phone_texts[index])
        phone_text = self.tokenize(None, phone_text)
        return phone_text

    def get_tokenized_src_char_text(self, index: int):
        char_text = self.tokenize(None, self.src_char_texts[index])
        char_text = self.tokenize(None, char_text)
        return char_text

    def __getitem__(self, index: int) -> SpeechToTextMSJointDatasetItem:
        s2t_dataset_item = super().__getitem__(index)
        src_tokens = None
        src_word_tokens = None
        src_phone_tokens = None
        src_char_tokens = None
        align_info = None
        src_lang_tag = None
        if self.src_texts is not None and self.src_dict is not None:
            src_tokens = self.get_tokenized_src_text(index)
            src_tokens = self.src_dict.encode_line(
                src_tokens, add_if_not_exist=False, append_eos=True
            ).long()
            if self.use_src_lang_id > 0:
                src_lang_tag = self.get_lang_tag_idx(
                    self.src_langs[index], self.src_dict
                )
        # processing src_word
        if self.src_word_texts is not None and self.src_word_dict is not None:
            src_word_tokens = self.get_tokenized_src_word_text(index)
            src_word_tokens = self.src_word_dict.encode_line(
                src_word_tokens, add_if_not_exist=False, append_eos=True
            ).long()
        # processing src_phone
        if self.src_phone_texts is not None and self.src_phone_dict is not None:
            src_phone_tokens = self.get_tokenized_src_phone_text(index)
            src_phone_tokens = self.src_phone_dict.encode_line(
                src_phone_tokens, add_if_not_exist=False, append_eos=True
            ).long()
        # processing src_char
        if self.src_char_texts is not None and self.src_char_dict is not None:
            src_char_tokens = self.get_tokenized_src_char_text(index)
            src_char_tokens = self.src_char_dict.encode_line(
                src_char_tokens, add_if_not_exist=False, append_eos=True
            ).long()
        tgt_lang_tag = None
        # processing aligns
        if self.speech_text_aligns is not None and len(self.speech_text_aligns) != 0:
            spch_txt_align_dict = json.loads(self.speech_text_aligns[index])
            spch_align = spch_txt_align_dict['align_spch']
            txt_align = spch_txt_align_dict['align_txt']
            if len(txt_align) != 0:
                audio_begin = torch.tensor([int(pair[0] / spch_align[-1][-1] * (s2t_dataset_item.source.size(0) / 8 + 1)) for pair in spch_align]).unsqueeze(0).long()
                audio_end = torch.tensor([int(pair[1] / spch_align[-1][-1] * (s2t_dataset_item.source.size(0) / 8 + 1)) for pair in spch_align]).unsqueeze(0).long()
                text_begin = torch.tensor([int(pair[0]) for pair in txt_align]).unsqueeze(0).long()
                text_end = torch.tensor([int(pair[1]) for pair in txt_align]).unsqueeze(0).long()
                align_info = torch.cat([audio_begin, audio_end, text_begin, text_end], dim=0).transpose(0, 1)
            else:
                align_info = "align info missing"

        if self.cfg.prepend_tgt_lang_tag_no_change:
            # prepend_tgt_lang_tag_no_change: modify prev_output_tokens instead
            tgt_lang_tag = self.get_lang_tag_idx(self.tgt_langs[index], self.tgt_dict)
        ali = None
        if self.alignment is not None:
            ali = torch.Tensor(self.alignment[index]).float()

        return SpeechToTextMSJointDatasetItem(
            index=index,
            source=s2t_dataset_item.source,
            target=s2t_dataset_item.target,
            src_txt_word_tokens=src_word_tokens,
            src_txt_phone_tokens=src_phone_tokens,
            src_txt_char_tokens=src_char_tokens,
            src_txt_tokens=src_tokens,
            spch_txt_align=align_info,
            tgt_lang_tag=tgt_lang_tag,
            src_lang_tag=src_lang_tag,
            tgt_alignment=ali,
        )

    def __len__(self):
        return self.n_samples

    def collater(self, samples: List[SpeechToTextMSJointDatasetItem]) -> Dict:
        s2t_out = super().collater(samples, return_order=True)
        if s2t_out == {}:
            return s2t_out
        net_input, order = s2t_out["net_input"], s2t_out["order"]

        if self.src_texts is not None and self.src_dict is not None:
            src_txt_tokens = fairseq_data_utils.collate_tokens(
                [x.src_txt_tokens for x in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            src_txt_lengths = torch.tensor(
                [x.src_txt_tokens.size()[0] for x in samples], dtype=torch.long
            )
            if self.use_src_lang_id > 0:
                src_lang_idxs = torch.tensor(
                    [s.src_lang_tag for s in samples], dtype=src_txt_tokens.dtype
                )
                if self.use_src_lang_id == 1:  # replace eos with lang_id
                    eos_idx = src_txt_lengths - 1
                    src_txt_tokens.scatter_(
                        1, eos_idx.view(-1, 1), src_lang_idxs.view(-1, 1)
                    )
                else:
                    raise NotImplementedError("Implementation is required")

            src_txt_tokens = src_txt_tokens.index_select(0, order)
            src_txt_lengths = src_txt_lengths.index_select(0, order)
            net_input["src_txt_tokens"] = src_txt_tokens
            net_input["src_txt_lengths"] = src_txt_lengths

        src_txt_word_tokens = None
        src_txt_word_lengths = None
        # add src word tokens
        if self.src_word_texts is not None and self.src_word_dict is not None:
            src_txt_word_tokens = fairseq_data_utils.collate_tokens(
                [x.src_txt_word_tokens for x in samples],
                self.src_word_dict.pad(),
                self.src_word_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            src_txt_word_lengths = torch.tensor(
                [x.src_txt_word_tokens.size()[0] for x in samples], dtype=torch.long
            )
            src_txt_word_tokens = src_txt_word_tokens.index_select(0, order)
            src_txt_word_lengths = src_txt_word_lengths.index_select(0, order)

        src_txt_phone_tokens = None
        src_txt_phone_lengths = None
        # add phone tokens
        if self.src_phone_texts is not None and self.src_phone_dict is not None:
            src_txt_phone_tokens = fairseq_data_utils.collate_tokens(
                [x.src_txt_phone_tokens for x in samples],
                self.src_phone_dict.pad(),
                self.src_phone_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            src_txt_phone_lengths = torch.tensor(
                [x.src_txt_phone_tokens.size()[0] for x in samples], dtype=torch.long
            )
            src_txt_phone_tokens = src_txt_phone_tokens.index_select(0, order)
            src_txt_phone_lengths = src_txt_phone_lengths.index_select(0, order)

        src_txt_char_tokens = None
        src_txt_char_lengths = None
        # add char tokens
        if self.src_char_texts is not None and self.src_char_dict is not None:
            src_txt_char_tokens = fairseq_data_utils.collate_tokens(
                [x.src_txt_char_tokens for x in samples],
                self.src_char_dict.pad(),
                self.src_char_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            src_txt_char_lengths = torch.tensor(
                [x.src_txt_char_tokens.size()[0] for x in samples], dtype=torch.long
            )
            src_txt_char_tokens = src_txt_char_tokens.index_select(0, order)
            src_txt_char_lengths = src_txt_char_lengths.index_select(0, order)

        align_pad = None
        align_lengths = None
        # add speech-text aligns
        if self.speech_text_aligns is not None and len(self.speech_text_aligns) != 0:
            # spch_txt_aligns = [x.spch_txt_align for x in samples]
            # list_order = order.tolist()
            # spch_txt_aligns = [spch_txt_aligns[i] for i in list_order]
            # align
            """
            别忘了处理空txt_align
            """
            align_info = [x.spch_txt_align if x.spch_txt_align != "align info missing" else None for x in samples]
            align_lengths = torch.LongTensor([a.size(0) if a is not None else 0 for a in align_info])
            align_maxlen = torch.max(align_lengths)
            align_pad = torch.full((len(align_info), align_maxlen, 4), -1, dtype=torch.long)
            for i in range(len(align_info)):
                if align_info[i] is not None:
                    tmp = torch.LongTensor(align_info[i])
                    if tmp.dim() == 2:
                        align_pad[i, :align_lengths[i], :] = tmp
            align_pad = align_pad.index_select(0, order)
            align_lengths = align_lengths.index_select(0, order)
            net_input["align_pad"] = align_pad
            net_input["align_lengths"] = align_lengths

        net_input["alignment"] = None
        if self.alignment is not None:
            max_len = max([s.tgt_alignment.size(0) for s in samples])
            alignment = torch.ones(len(samples), max_len).float()
            for i, s in enumerate(samples):
                cur_len = s.tgt_alignment.size(0)
                alignment[i][:cur_len].copy_(s.tgt_alignment)
            net_input["alignment"] = alignment.index_select(0, order)

        if self.tgt_texts is not None and samples[0].tgt_lang_tag is not None:
            for i in range(len(samples)):
                net_input["prev_output_tokens"][i][0] = samples[order[i]].tgt_lang_tag

        out = {
            "id": s2t_out["id"],
            "net_input": net_input,
            "target": s2t_out["target"],
            "target_lengths": s2t_out["target_lengths"],
            "target_src_word": src_txt_word_tokens,
            "target_lengths_src_word": src_txt_word_lengths,
            "target_word": src_txt_word_tokens,
            "target_lengths_word": src_txt_word_lengths,
            "target_phone": src_txt_phone_tokens,
            "target_lengths_phone": src_txt_phone_lengths,
            "target_char": src_txt_char_tokens,
            "target_lengths_char": src_txt_char_lengths,
            "ntokens": s2t_out["ntokens"],
            "nsentences": len(samples),
        }
        return out


class SpeechToTextMSJointDatasetCreator(SpeechToTextDatasetCreator):
    KEY_ALIGN = "align"
    KEY_SRC_PHONE_TEXT = "src_text"
    KEY_SRC_CHAR_TEXT = "src_char_text"
    KEY_SRC_WORD_TEXT = "src_word_text"
    KEY_SPEECH_TEXT_ALIGN = "speech_text_align"
    DEFAULT_SRC_WORD_TEXT = DEFAULT_SRC_PHONE_TEXT = DEFAULT_SRC_CHAR_TEXT = DEFAULT_SPEECH_TEXT_ALIGN = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TMSJointDataConfig,
        tgt_dict,
        src_dict,
        src_word_dict,
        src_phone_dict,
        src_char_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        src_word_pre_tokenizer,
        src_word_bpe_tokenizer,
        append_eos,
        use_src_lang_id,
    ) -> SpeechToTextMSJointDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        src_phone_texts = [s.get(cls.KEY_SRC_PHONE_TEXT, cls.DEFAULT_SRC_PHONE_TEXT) for s in samples]
        src_char_texts = [s.get(cls.KEY_SRC_CHAR_TEXT, cls.DEFAULT_SRC_CHAR_TEXT) for s in samples]
        src_word_texts = [s.get(cls.KEY_SRC_WORD_TEXT, cls.DEFAULT_SRC_WORD_TEXT) for s in samples]
        speech_text_aligns = [s.get(cls.KEY_SPEECH_TEXT_ALIGN) for s in samples if s.get(cls.KEY_SPEECH_TEXT_ALIGN) is not None]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_alignment = None
        if cls.KEY_ALIGN in samples[0].keys():
            tgt_alignment = [s[cls.KEY_ALIGN] for s in samples]
        return SpeechToTextMSJointDataset(
            split_name,
            is_train_split,
            cfg,
            audio_paths,
            n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            src_word_texts=src_word_texts,
            src_phone_texts=src_phone_texts,
            src_char_texts=src_char_texts,
            speech_text_aligns=speech_text_aligns,
            speakers=speakers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            src_dict=src_dict,
            src_word_dict=src_word_dict,
            src_phone_dict=src_phone_dict,
            src_char_dict=src_char_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            src_word_pre_tokenizer=src_word_pre_tokenizer,
            src_word_bpe_tokenizer=src_word_bpe_tokenizer,
            append_eos=append_eos,
            alignment=tgt_alignment,
            use_src_lang_id=use_src_lang_id,
        )

    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2TMSJointDataConfig,
        split: str,
        tgt_dict,
        src_dict,
        src_word_dict,
        src_phone_dict,
        src_char_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        src_word_pre_tokenizer,
        src_word_bpe_tokenizer,
        append_eos: bool,
        use_src_lang_id: int,
    ) -> SpeechToTextMSJointDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            tgt_dict,
            src_dict,
            src_word_dict,
            src_phone_dict,
            src_char_dict,
            pre_tokenizer,
            bpe_tokenizer,
            src_pre_tokenizer,
            src_bpe_tokenizer,
            src_word_pre_tokenizer,
            src_word_bpe_tokenizer,
            append_eos,
            use_src_lang_id,
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2TMSJointDataConfig,
        splits: str,
        tgt_dict,
        src_dict,
        src_word_dict,
        src_phone_dict,
        src_char_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        src_word_pre_tokenizer,
        src_word_bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        append_eos: Optional[bool] = True,
        use_src_lang_id: Optional[int] = 0,
    ) -> SpeechToTextMSJointDataset:
        datasets = [
            cls._from_tsv(
                root,
                cfg,
                split,
                tgt_dict,
                src_dict,
                src_word_dict,
                src_phone_dict,
                src_char_dict,
                is_train_split,
                pre_tokenizer,
                bpe_tokenizer,
                src_pre_tokenizer,
                src_bpe_tokenizer,
                src_word_pre_tokenizer,
                src_word_bpe_tokenizer,
                append_eos=append_eos,
                use_src_lang_id=use_src_lang_id,
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
