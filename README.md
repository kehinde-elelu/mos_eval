# Baseline system of the AudioMOS Challenge-Track1
The system implemented in this repository serves as the baseline of track 1 of the [AudioMOS Challenge](https://sites.google.com/view/voicemos-challenge/audiomos-challenge-2025) , leveraging CLAP to predict OVERALL MUSICAL QUALITY and TEXTUAL ALIGNMENT with input text descriptions of generated music.

Author: 

- Cheng Liu (College of Computer Science, Nankai University) liucheng_hlt@mail.nankai.edu.cn
- Hui Wang (College of Computer Science, Nankai University) wanghui_hlt@mail.nankai.edu.cn

## Training Phase (Phase 1)

During the training phase, the training set and the developement set are released. In the following, we demonstrate how to use the baseline model to make predictions on the development set of [MusicEval](https://arxiv.org/abs/2501.10811), and generate a results file that can be submitted to the CodaLab platform.

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/NKU-HLT/MusicEval-baseline.git
    cd MusicEval-baseline
    ```

2. **Set Up the Environment**

    ```bash
    conda create -n musiceval python=3.8.18
    conda activate musiceval

    # Install additional requirements
    pip install -r requirements.txt
    ```

### Download Pretrained Model

Please download the CLAP checkpoint [here](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt), which is pretrained on music+AudioSet+LAION-Audio-630k. Put the checkpoint in `./upstream/`.

### Data Preparation

The dataset for track1 will be released on CodaBench soon. 

### Training

Firstly, make sure you have already prepare the dataset and one pretrained CLAP model. All the required files are like below:

```
MusicEval-baseline/
|_README.md
|_code/
  |_mos_track1.py
  |_predict.py
  |_predict_noGT.py
  |_utils.py
|_data/
  |_MusicEval-phase1/
|_upstream/
  |_music_audioset_epoch_15_esc_90.14.pt

```

Now you can start training the baseline system by 

```shell
cd ./code
python mos_track1.py --datadir ../data/MusicEval-phase1  --expname EXPNAME
```

Once the training has finished, the best checkpoint can be found in `./track1_exp/EXPNAME/` directory, named `best_ckpt_NUM.ckpt`, where `NUM` is the best epoch.

### Predicting

To run inference on development set using the checkpoint, run:

```shell
python predict.py
```

This will generate `answer_track1.txt`, includes prediction scores for dev-set samples.  Also, it will print the utterance-level and system-level metrics in both OVERALL QUALITY and TEXTUAL ALIGNMENT dimensions, like below:

```
==========UTTERANCE===========
======OVERALL QUALITY=======
[UTTERANCE] Test error= 0.617519
[UTTERANCE] Linear correlation coefficient= 0.690845
[UTTERANCE] Spearman rank correlation coefficient= 0.688106
[UTTERANCE] Kendall Tau rank correlation coefficient= 0.514346
======TEXTUAL ALIGNMENT=======
[UTTERANCE] Test error= 0.593582
[UTTERANCE] Linear correlation coefficient= 0.580330
[UTTERANCE] Spearman rank correlation coefficient= 0.542522
[UTTERANCE] Kendall Tau rank correlation coefficient= 0.393303
==========SYSTEM===========
======OVERALL QUALITY=======
[SYSTEM] Test error= 0.386326
[SYSTEM] Linear correlation coefficient= 0.801622
[SYSTEM] Spearman rank correlation coefficient= 0.776355
[SYSTEM] Kendall Tau rank correlation coefficient= 0.586207
======TEXTUAL ALIGNMENT=======
[SYSTEM] Test error= 0.232161
[SYSTEM] Linear correlation coefficient= 0.746149
[SYSTEM] Spearman rank correlation coefficient= 0.720197
[SYSTEM] Kendall Tau rank correlation coefficient= 0.507389

```

### Submission to CodaLab

The submission format of the CodaLab competition platform is a zip file (can be any name) containing a text file called `answer.txt` (this naming is a **MUST**).  

The competition link will be released soon.

## **Acknowledgments**

This project builds upon prior work from the [nii-yamagishilab/mos-finetune-ssl](https://github.com/nii-yamagishilab/mos-finetune-ssl) repository. We thank them for their contributions! 

## **Citation**

If you use the repo in your research, please cite us as follows:

```bibtex
@INPROCEEDINGS{10890307,
  author={Liu, Cheng and Wang, Hui and Zhao, Jinghua and Zhao, Shiwan and Bu, Hui and Xu, Xin and Zhou, Jiaming and Sun, Haoqin and Qin, Yong},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={MusicEval: A Generative Music Dataset with Expert Ratings for Automatic Text-to-Music Evaluation}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Costs;Signal processing;Predictive models;Acoustics;Quality assessment;Complexity theory;Speech processing;mean opinion score;text-to-music generation;automatic quality assessment},
  doi={10.1109/ICASSP49660.2025.10890307}}

```