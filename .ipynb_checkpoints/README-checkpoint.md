<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/BodgeDook/Disaster-Tweets">
    <img src="./docs_src/preview.jpg" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">Natural Language Processing with Disaster Tweets</h3>
</p>

<p align="center">
  <a href="https://github.com/BodgeDook/"><b>Dmitry Zlobin</b></a> · <a href="https://github.com/KamenevIvan/"><b>Ivan Kamenev</b></a> · <a href="https://github.com/703lovelost/"><b>Aleksey Spirkin</b></a>
  <br />
  Deep Robotics Institute
  <br />
  Novosibirsk State University
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of contents</summary>
  <ol>
    <li><a href="#About">About</a></li>
    <li><a href="#Preprocessing">Preprocessing</a></li>
    <li><a href="#Training-and-inference">Training and inference</a></li>
  </ol>
</details>

## About

This project was formed to participate in the Disaster Tweets Kaggle competition.

We use <a href="https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion">`bhadresh-savani/distilbert-base-uncased-emotion`</a> to search the tweets' pattern.
<br />
Check the competition details on the link <a href="https://www.kaggle.com/competitions/nlp-getting-started">here</a>.

## Preprocessing

There's no need for any preprocessing, everything is already prepared for training and inference. The dataset expansion is welcome.

Just make sure the requirements is satisfied:
```
    pip install -r requirements.txt
```

## Training and inference

To run the training:
```
  # Arguments and their default values:
  # --exp_name - run_%Y%m%d_%H%M%S
  # --train_csv - data/train_modified.csv
  # --model_name - bhadresh-savani/distilbert-base-uncased-emotion
  # --lr - 5e-5
  # --epochs - 3
  # --batch_size - 16
  # --output_base - outputs

  python3 train.py --exp_name EXPERIMENT_FOLDER_NAME --epochs NUM_EPOCHS --batch_size NUM_BATCH
```

To run the validation:
```
  # Arguments and their default values:
  # (Required) --exp_dir - no default value
  # --test_csv - data/test_modified.csv

  python3 validate.py --exp_dir EXPERIMENT_FOLDER_NAME
```