# end-**T**o-end **A**daptive **L**ocal **L**earning (**TALL**)
## MoE with MultVAE as Expert
<!-- Inspired by LOCA idea and Fight Mainstream Bias via Local Fine Tuning. <br>

Original LOCA GitHub link is [here](https://github.com/jin530/LOCA).

Original LFT and EnLFT code is [here](https://github.com/Zziwei/Measuring-Mitigating-Mainstream-Bias).

This is the official code with preprocessed datasets for the WSDM 2021 paper: [`Local Collaborative Autoencoders`.](https://arxiv.org/abs/2103.16103)

This is the offical code with preprocessed datasets for the WSDM 2022 paper: [`Fighting Mainstream Bias in Recommender Systems via LocalFine Tuning`.](https://dl.acm.org/doi/pdf/10.1145/3488560.3498427)

The slides can be found [here](https://www.slideshare.net/ssuser1f2162/local-collaborative-autoencoders-wsdm2021). -->

---

## Dataset

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Dataset</th>
    <th class="tg-dvpl"># Users</th>
    <th class="tg-dvpl"># Items</th>
    <th class="tg-dvpl"># Ratings</th>
    <th class="tg-dvpl">Sparsity</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">ML1M</td>
    <td class="tg-dvpl">6,040</td>
    <td class="tg-dvpl">3,952</td>
    <td class="tg-dvpl">1,000,209</td>
    <td class="tg-dvpl">95.81%</td>
  </tr>
  <tr>
    <td class="tg-0pky">Amazon CDs & Vinyl</td>
    <td class="tg-dvpl">24,179</td>
    <td class="tg-dvpl">27,602</td>
    <td class="tg-dvpl">4,543,369</td>
    <td class="tg-dvpl">99.32%</td>
  </tr>
  <tr>
    <td class="tg-0pky">Yelp</td>
    <td class="tg-dvpl">25,677</td>
    <td class="tg-dvpl">25,815</td>
    <td class="tg-dvpl">731,671</td>
    <td class="tg-dvpl">99.89%</td>
  </tr>
</tbody>
</table>

The table contains the information of original non-preprocessed datasets.
<br>
<br>
I uploaded one public benchmark datasets code sample here: MovieLens 1M (ML1M). You can find other datasets and run them in this code. We convert all explicit ratings to binary values, whether the ratings are observed or missing, some example datasets are listed in the table above.
<br>
<br>

You can get the original datasets from the following links:
<!-- Movielens -->
Movielens: https://grouplens.org/datasets/movielens/

<!-- Amazon review -->
Amazon Review Data: https://nijianmo.github.io/amazon/

<!-- Yelp -->
Yelp 2015: https://github.com/hexiangnan/sigir16-eals/tree/master/data

---

## Basic Usage
- Change the experimental settings in `main_config.cfg` and the model hyperparameters in `model_config`. </br>
- Run `main.py` to train and test models. </br>
- Command line arguments are also acceptable with the same naming in configuration files. (Both main/model config)

For example: ```python main.py --model_name MultVAE --lr 0.001```

## Running LOCA
Before running LOCA, you need (1) user embeddings to find local communities and (2) the global model to cover users who are not considered by local models. </br>

1. Run single MultVAE to get user embedding vectors and the global model: 

`python main.py --model_name MultVAE` 

2. Train LOCA with the specific backbone model:

`python main.py --model_name LOCA_VAE` 

## Running TALL (MoE)
`python main.py --model_name MOE`

---

## Requirements
- Python 3.7 or higher
- Torch 1.5 or higher

<!-- ## Citation
cited papaer:
```
@inproceedings{DBLP:conf/wsdm/ChoiJLL21,
  author    = {Minjin Choi and
               Yoonki Jeong and
               Joonseok Lee and
               Jongwuk Lee},
  title     = {Local Collaborative Autoencoders},
  booktitle = {{WSDM} '21, The Fourteenth {ACM} International Conference on Web Search
               and Data Mining, Virtual Event, Israel, March 8-12, 2021},
  pages     = {734--742},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3437963.3441808},
  doi       = {10.1145/3437963.3441808},
  timestamp = {Wed, 07 Apr 2021 16:17:44 +0200},
  biburl    = {https://dblp.org/rec/conf/wsdm/ChoiJLL21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{zhu2022fighting,
  title={Fighting mainstream bias in recommender systems via local fine tuning},
  author={Zhu, Ziwei and Caverlee, James},
  booktitle={Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
  pages={1497--1506},
  year={2022}
}
``` -->
