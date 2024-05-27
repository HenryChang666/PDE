# DPA
Improving End-to-End Speech Translation with Progressive Dual-encoding

## install
We conducted our experiments on fairseq-0.12.2. To install the dependencies required for the experiment, run the following command:
```
pip install -r requirements.txt
```
Insert speech_text_ms_joint_to_text into the examples directory of fairseq, and then move the examples directory to the same directory level as fairseq.

## Prepare Data
#### Prepare MuST-C data set
-   Please follow the data preparation in the [S2T example](https://github.com/pytorch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md)
-   Convert source text to its char representation and place it in the "src_char_text" column of TSV file
-   Convert source text to its phoneme representation and place it in the "src_text" column of TSV file
-   Place the source text in the "src_word_text" column of TSV file
-   Apply MFA for forced alignment and record the alignment information in JSON format under the 'speech_text_align' column in the TSV file.
A snapshot of the final TSV file is shown below:
```
id	audio	n_frames	tgt_text	speaker	src_word_text	src_char_text	src_text	speech_text_align
ted_1_0	/mnt/zhangrunlai/mustc/en-de/fbank80.zip:2550382743:51328	160	Hinter mir war gar keine Autokolonne.	spk.1	There was no motorcade back there.	t h e r e | w a s | n o | m o t o r c a d e | b a c k | t h e r e .	▁DH EH1 R ▁W AA1 Z ▁N OW1 ▁M OW1 T ER0 K EY2 D ▁B AE1 K ▁DH EH1 R	{"align_spch": [[0.0, 0.16], [0.16, 0.31], [0.31, 0.47], [0.47, 1.1], [1.1, 1.31], [1.36, 1.62]], "align_txt": [[0, 3], [3, 6], [6, 8], [8, 15], [15, 18], [18, 21]], "word_token": [[54], [23], [124], [2387, 169, 2993], [139], [54]]}
ted_1_1	/mnt/zhangrunlai/mustc/en-de/fbank80.zip:16042387858:438208	1369	Haben Sie schon mal vom Phantomschmerz gehört? (Lachen) Wir saßen in einem gemieteten Ford Taurus.	spk.1	(Laughter) You've heard of phantom limb pain? (Laughter)	( l a u g h t e r ) | y o u ' v e | h e a r d | o f | p h a n t o m | l i m b | p a i n ? | ( l a u g h t e r )	▁VOICE ▁Y UW1 V ▁HH ER1 D ▁AH1 V ▁F AE1 N T AH0 M ▁L IH1 M ▁P EY1 N ▁VOICE	{"align_spch": [[0.0, 0.05], [0.05, 0.16], [0.16, 0.32], [0.32, 0.46], [0.46, 0.89], [0.89, 1.21], [1.21, 1.68], [1.68, 13.70994]], "align_txt": [[0, 1], [1, 4], [4, 7], [7, 9], [9, 15], [15, 18], [18, 21], [21, 22]], "word_token": [[724, 530, 2316, 294, 501, 708], [17, 9, 78], [554], [10], [5436], [2919], [1112], [724, 530, 2316, 294, 501, 708]]}
ted_1_2	/mnt/zhangrunlai/mustc/en-de/fbank80.zip:36784304945:142208	444	Es war Zeit zum Abendessen und wir hielten Ausschau nach einem Restaurant.	spk.1	It was dinnertime, and we started looking for a place to eat.	i t | w a s | d i n n e r t i m e , | a n d | w e | s t a r t e d | l o o k i n g | f o r | a | p l a c e | t o | e a t .	▁IH1 T ▁W AA1 Z ▁D IH1 N ER0 T AY2 M ▁AH0 N D ▁W IY1 ▁S T AA1 R T AH0 D ▁L UH1 K IH0 NG ▁F AO1 R ▁AH0 ▁P L EY1 S ▁T UW1 ▁IY1 T	{"align_spch": [[0.0, 0.15], [0.15, 0.5], [0.5, 1.21], [1.77, 2.46], [2.95, 3.1], [3.1, 3.39], [3.39, 3.63], [3.63, 3.72], [3.72, 3.75], [3.75, 4.04], [4.04, 4.12], [4.12, 4.46]], "align_txt": [[0, 2], [2, 5], [5, 12], [12, 15], [15, 17], [17, 24], [24, 29], [29, 32], [32, 33], [33, 37], [37, 39], [39, 41]], "word_token": [[18], [23], [2159, 1164], [12], [19], [218], [292], [24], [11], [247], [8], [879]]}
ted_1_3	/mnt/zhangrunlai/mustc/en-de/fbank80.zip:13995182400:17088	53	Wir waren auf der I-40.	spk.1	We were on I-40.	w e | w e r e | o n | i - 4 0 .	▁W IY1 ▁W ER0 ▁AA1 N ▁AY1 ▁F AO1 R T IY0	{"align_spch": [[0.0, 0.06], [0.06, 0.23], [0.23, 0.34], [0.34, 0.49], [0.49, 0.55]], "align_txt": [[0, 2], [2, 4], [4, 6], [6, 7], [7, 12]], "word_token": [[19], [87], [28], [3380], [909]]}
```

```
-   Prepare phoneme dictionary and save to $MANIFEST_ROOT as [src_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/src_dict.txt)
#### Prepare WMT text data
-   [Download wmt data](https://github.com/pytorch/fairseq/blob/main/examples/translation/prepare-wmt14en2de.sh)
-   Convert source text (English) into phoneme representation as above
-   Generate binary parallel files with "fairseq-preprocess" from fairseq for training and validation. The source input is English phoneme representation and the target input is German sentencepiece token .  The output is saved under $parallel_text_data

## Training
The model is trained with 8 v100 GPUs.

#### Download pretrained models
-    [pretrain_encoder](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)
-    [pretrain_nmt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/checkpoint_mt.pt)

#### Training scripts
- Jointly trained model from scratch
```bash
python train.py ${MANIFEST_ROOT} \
    --save-dir ${save_dir} \
    --num-workers 8 \
    --task speech_text_joint_to_text \
    --arch dualinputs2ttransformer_s \
    --user-dir examples/speech_text_joint_to_text \
    --max-epoch 100 --update-mix-data \
    --optimizer adam --lr-scheduler inverse_sqrt \
    --lr 0.001 --update-freq 4 --clip-norm 10.0 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --label-smoothing 0.1 --max-tokens 10000 --max-tokens-text 10000 \
    --max-positions-text 400 --seed 2 --speech-encoder-layers 12 \
    --text-encoder-layers 6 --encoder-shared-layers 6 --decoder-layers 6 \
    --dropout 0.1 --warmup-updates 20000  \
    --text-sample-ratio 0.25 --parallel-text-data ${parallel_text_data} \
    --text-input-cost-ratio 0.5 --enc-grad-mult 2.0 --add-speech-eos \
    --log-format json --langpairs en-de --noise-token '"'"'▁NOISE'"'"' \
    --mask-text-ratio 0.0 --max-tokens-valid 20000 --ddp-backend no_c10d \
    --log-interval 100 --data-buffer-size 50 --config-yaml config.yaml \
    --keep-last-epochs 10
```
- Jointly trained model with good initialization, cross attentive loss and online knowledge distillation
```bash
python train.py ${MANIFEST_ROOT} \
    --save-dir ${save_dir} \
    --num-workers 8 \
    --task speech_text_joint_to_text \
    --arch dualinputs2ttransformer_m \
    --user-dir examples/speech_text_joint_to_text \
    --max-epoch 100 --update-mix-data \
    --optimizer adam --lr-scheduler inverse_sqrt \
    --lr 0.002 --update-freq 4 --clip-norm 10.0 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --guide-alpha 0.8 --disable-text-guide-update-num 5000 \
    --label-smoothing 0.1 --max-tokens 10000 --max-tokens-text 10000 \
    --max-positions-text 400 --seed 2 --speech-encoder-layers 12 \
    --text-encoder-layers 6 --encoder-shared-layers 6 --decoder-layers 6 \
    --dropout 0.1 --warmup-updates 20000 --attentive-cost-regularization 0.02 \
    --text-sample-ratio 0.25 --parallel-text-data ${parallel_text_data} \
    --text-input-cost-ratio 0.5 --enc-grad-mult 2.0 --add-speech-eos \
    --log-format json --langpairs en-de --noise-token '"'"'▁NOISE'"'"' \
    --mask-text-ratio 0.0 --max-tokens-valid 20000 --ddp-backend no_c10d \
    --log-interval 100 --data-buffer-size 50 --config-yaml config.yaml \
    --load-pretrain-speech-encoder ${pretrain_encoder} \
    --load-pretrain-decoder ${pretrain_nmt} \
    --load-pretrain-text-encoder-last ${pretrain_nmt} \
    --keep-last-epochs 10
```

## Evaluation
```bash
python ./fairseq_cli/generate.py \
        ${MANIFEST_ROOT} \
        --task speech_text_joint_to_text \
        --max-tokens 25000 \
        --nbest 1 \
        --results-path ${infer_results} \
        --batch-size 512 \
        --path ${model} \
        --gen-subset tst-COMMON_st \
        --config-yaml config.yaml \
        --scoring sacrebleu \
        --beam 5 --lenpen 1.0 \
        --user-dir examples/speech_text_joint_to_text \
        --load-speech-only
```

## Results (Joint training with initialization + CAR + online KD)
|Direction|En-De | En-Es | En-Fr |
|---|---|---|---|
|BLEU|27.4| 31.2 | 37.6 |
|checkpoint | [link](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/checkpoint_ave_10.pt) |[link](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_es/checkpoint_ave_10.pt)|[link](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_fr/checkpoint_ave_10.pt)|
