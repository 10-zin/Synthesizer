# Synthesizer
A PyTorch implementation of the paper : [SYNTHESIZER: Rethinking Self-Attention in Transformer Models](https://arxiv.org/abs/2005.00743) - Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, Che Zheng

The paper majorly proposes two efficient variants of scaled dot product attention in the regular Transformers.

The snapshot from the paper below perfectly illustrates the difference.

<p align="center">
<img src="https://github.com/10-zin/Synthesizer/blob/master/images/Synthesizer.jpeg" width="700">
</p>

# Variants

This repository currently consists of the implementations for the following variants:
1. Vanilla Attention (regular scaled dot product attention based Transformer)
2. Dense Attention
3. Factorized Dense Attention
4. Random Attention
5. Factorized Random Attention

# Usage

## WMT'16 Multimodal Translation: de-en

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).
### 0) Create venv, install requirements and move to the synth directory
```bash
python3 -m venv synth-env
source synth-env/bin/activate
pip install -r requirements.txt
cd synth/
```
### 1) Download the spacy language model.
```bash
# conda install -c conda-forge spacy 
python -m spacy download en
python -m spacy download de
```

### 2) Preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

### 3) Train the model
```bash
python train.py -data_pkl m30k_deen_shr.pkl -log log_dense_1 -embs_share_weight -proj_share_weight -label_smoothing -save_model trained_dense_1 -b 8 -warmup 128000 -n_head 2 -n_layers 2 -attn_type dense -epoch 25
```

### 4) Test the model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```

# Comparisons

<p align="center">
<img src="https://github.com/10-zin/Synthesizer/blob/master/images/loss-1.png" width="400">
<img src="https://github.com/10-zin/Synthesizer/blob/master/images/ppl-1.png" width="400">
</p>

<p align="center">

</p>
