# Baseline system of the AudioMOS Challenge-Track1

------

Author: 

- Cheng Liu (College of Computer Science, Nankai University) liucheng_hlt@mail.nankai.edu.cn
- Hui Wang (College of Computer Science, Nankai University) wanghui_hlt@mail.nankai.edu.cn

## Training Phase (Phase 1)

During the training phase, the training set and the developement set are released. In the following, we demonstrate how to use the baseline model in [MusicEval](https://arxiv.org/abs/2501.10811) to make predictions on the development set to generate a results file that can be submitted to the CodaLab platform.

### Upstream Model Preparation

Please download the CLAP model pretrained on music+AudioSet+LAION-Audio-630k. You can download the pretrained model  [here](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt) and put it in `./upstream/`.

### Data Preparation

Please download the challenge dataset for track1 here: [MusicEval-phase1](https://drive.google.com/drive/folders/1Qt3B5dIaqJnjm1NlvpV1uQFx_OXDkD9S?usp=drive_link) , unzip and put it in `./data/`. 





### Training

First, make sure you already have the dataset and one pretrained CLAP model. All the required files like below:

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

Now you can start training phase by 

```shell
cd ./code
python mos_track1.py --datadir ./data  --expname EXPNAME
```

Once the training has finished, the best checkpoint can be found in `./track1_exp/EXPNAME/` directory, named `best_NUM.ckpt`. The tensorboard log can be found in `./track1_log/EXP_NAME/` directory.

### Predicting

To run inference on development set using the checkpoint, run:

```shell
python predict.py --datadir ./data --expname EXPNAME
```

That will generate `answer_track1.txt`, includes prediction scores for dev-set samples.  Also, it will print the utterance-level and system-level metrics in both OVERALL QUALITY and TEXTUAL ALIGNMENT, like below:

```
==========UTTERANCE===========
======OVERALL QUALITY=======
[UTTERANCE] Test error= 0.407076
[UTTERANCE] Linear correlation coefficient= 0.752050
[UTTERANCE] Spearman rank correlation coefficient= 0.760752
[UTTERANCE] Kendall Tau rank correlation coefficient= 0.570974
======TEXTUAL ALIGNMENT=======
[UTTERANCE] Test error= 0.507618
[UTTERANCE] Linear correlation coefficient= 0.593813
[UTTERANCE] Spearman rank correlation coefficient= 0.581332
[UTTERANCE] Kendall Tau rank correlation coefficient= 0.422476
==========SYSTEM===========
======OVERALL QUALITY=======
[SYSTEM] Test error= 0.186899
[SYSTEM] Linear correlation coefficient= 0.878047
[SYSTEM] Spearman rank correlation coefficient= 0.882266
[SYSTEM] Kendall Tau rank correlation coefficient= 0.704433
======TEXTUAL ALIGNMENT=======
[SYSTEM] Test error= 0.143016
[SYSTEM] Linear correlation coefficient= 0.783773
[SYSTEM] Spearman rank correlation coefficient= 0.795567
[SYSTEM] Kendall Tau rank correlation coefficient= 0.596059
```

### Submission to CodaLab

The submission format of the CodaLab competition platform is a zip file (can be any name) containing a text file called `answer.txt` (this naming is a **MUST**).  

You may submit main-track predictions only, or main-track and ood-track predictions together.  Since the main track is mandatory and the OOD track is optional, you may NOT submit OOD predictions by themselves -- this will fail to validate on CodaLab.

You can prepare a submission file for CodaLab like this:

```
cat answer_main.txt answer_ood.txt > answer.txt
zip -j anyname.zip answer.txt
```

Then this zip file is ready to be submitted!