# From Generality to Mastery: Composer-Style Symbolic Music Generation via Large-Scale Pre-training

This is the official repository of the AIMC 2025 paper "From Generality to Mastery: Composer-Style Symbolic Music Generation via Large-Scale Pre-training."

[Paper](https://arxiv.org/abs/2506.17497) | [Demo page](https://generality-mastery.github.io/) | [Model weights & Dataset](https://huggingface.co/Itsuki-music/Generality_to_Mastery)

## Environment
* **Python 3.8** recommended
* Install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start
### Composer-style Conditioned Symbolic Music Generation
**Generate music**

Generate music with a particular composer, set to 'None' for generality model
```bash
python3 inference_score.py \
    --configuration=config/score_top5_finetune.yaml \
    --representation=remi \
    --output_dir=your/sample/path \
    --composer=chopin
```
Generate music with all four composers trained on
```bash
python3 inference_score.py \
    --configuration=config/score_top5_finetune.yaml \
    --representation=remi \
    --output_dir=your/sample/path \
    --all_composer=True
```

## Train a model by yourself
We take training a **mastery (two-stage) model** as an example.

1. Use the provided [events](https://huggingface.co/Itsuki-music/Generality_to_Mastery) directly or convert MIDI to events using the provided [scripts](https://github.com/AndyWeasley2004/Generality-to-Mastery/tree/main/representations#readme).

2. Pre-train a generality model
```bash
python3 train_score.py \
    --configuration=config/score_pretrain.yaml
```

3. Fine-tune with the composer's data and add adapter modules
```bash
python3 finetune_adapter.py \
    --configuration=config/score_top5_finetune.yaml
```

## Dataset
We open-sourced the [dataset](https://huggingface.co/Itsuki-music/Generality_to_Mastery/tree/main/full) we use for training.
* [pretrain_score.zip](https://huggingface.co/Itsuki-music/Generality_to_Mastery/blob/main/full/pretrain_score.zip) contains all genres of data for training. The folk and pop genres come from various public-domain datasets, and you can find the original dataset with the prefix of the filenames. The classical data combines a large number of datasets, which disucssed in more details in the [paper]().
* [finetune_top4_selected_new.zip](https://huggingface.co/Itsuki-music/Generality_to_Mastery/blob/main/full/finetune_top4_selected_new.zip) contains the manually checked, organized, and augmented data of Bach, Beethoven, Chopin, and Mozart's works.
* [huggingface_data.zip](https://huggingface.co/Itsuki-music/Generality_to_Mastery/blob/main/full/huggingface_data.zip) contains data we use to train the evaluation classifier. It comes from a [public classical dataset](https://huggingface.co/datasets/drengskapur/midi-classical-music).

## Citation
If you find this project useful, please cite our paper:
```
@inproceedings{generalitymastery2025,
  author = {Mingyang Yao and Ke Chen},
  title = {From Generality to Mastery: Composer-Style Symbolic Music Generation via Large-Scale Pre-training},
  booktitle={Proceedings of the AI Music Creativity, {AIMC}},
  year = {2025}
}
```
