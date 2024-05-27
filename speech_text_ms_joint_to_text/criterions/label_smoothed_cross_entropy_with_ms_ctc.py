# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.data.data_utils import lengths_to_mask
@dataclass
class LabelSmoothedCrossEntropyWithCtcCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    ctc_weight: float = field(default=1.0, metadata={"help": "weight for CTC loss"})
    zero_infinity: bool = field(default=True, metadata={"help": "zero inf loss when source length <= target length"})
    post_process: str = field(default="letter", metadata={"help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"})


@register_criterion(
    "label_smoothed_cross_entropy_with_ms_ctc",
    dataclass=LabelSmoothedCrossEntropyWithCtcCriterionConfig,
)
class LabelSmoothedCrossEntropyWithCtcCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        post_process="letter",
        ctc_weight=0.0
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size=0
        )
        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.report_accuracy = True
        self.ctc_weight = ctc_weight
        if self.ctc_weight > 0:
#            assert getattr(task, "src_dict", None) is not None, "CTC need a source dictionary."
            self.zero_infinity = True
            self.post_process = post_process

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if self.ctc_weight > 0 :
            # ctc_loss = self.compute_ctc_loss(model, sample, net_output, reduce)
            ctc_loss = sum(self.compute_multi_scale_ctc_loss(model, sample, net_output, reduce))
            logging_output["ctc_loss"] = utils.item(ctc_loss.data)
            loss = (1 - self.ctc_weight) * loss + ctc_loss
            logging_output["loss"] = utils.item(loss.data) if reduce else loss.data


        return loss, sample_size, logging_output

    def compute_multi_scale_ctc_loss(self, model, sample, net_output, reduce: bool):
        if self.task.__class__.__name__ == 'MultiScaleASRTask':
            scale_dict = {"char": "_char", "phone": "_phone", "word": "_word"}
        elif self.task.__class__.__name__ == 'MSTranslationTask':
            scale_dict = {}
            if self.task.source_word_dictionary is not None:
                scale_dict["src_word"] = "_src_word"
            if self.task.target_word_dictionary is not None:
                scale_dict["tgt_word"] = "_tgt_word"

        ctc_loss = []
        for key in scale_dict:
            ctc_lprobs, ctc_lens = model.get_ctc_output(net_output, sample, scale_dict[key])
            ctc_tgt, ctc_tgt_lens = model.get_ctc_target(sample, scale_dict[key])
            ctc_tgt_mask = lengths_to_mask(ctc_tgt_lens)
            ctc_tgt_flat = ctc_tgt.masked_select(ctc_tgt_mask)
            reduction = "sum" if reduce else "none"
            loss = (
                    F.ctc_loss(
                        ctc_lprobs,
                        ctc_tgt_flat,
                        ctc_lens,
                        ctc_tgt_lens,
                        reduction=reduction,
                        zero_infinity=True,
                    )
                    * self.ctc_weight
            )
            ctc_loss.append(loss)
        return ctc_loss


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        trans_loss_sum = utils.item(
            sum(log.get("trans_loss", 0) for log in logging_outputs)
        )
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "trans_loss", trans_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ctc_loss",
            ctc_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return True
