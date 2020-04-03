# Long & Short-term Preference Model for Recommendation

In the era of information explosion, recommender systems have become an indispensable part of modern life. Users' decisions usually are determined by long-term and short-term preferences together. However, most of the existing work usually study these two requirements separately. In this paper, we attempt to build the bridge between long-term preference and short-term preference. We propose a Long & Short-term Preference Model (LSPM), which incorporates LSTM and self-attention mechanism to learn the short-term preference and jointly model long-term preference by a neural latent factor model from historical interactions. We conduct experiments to demonstrate the effectiveness of LSPM on three public datasets. Compared with the state-of-the-art methods, LSPM gets a massive improvement in HR@10 and NDCG@10,  which relatively increased by %3.875 and %6.363.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/chenjie04/LSPM.git
    $ cd LSPM/
    $ sudo pip3 install -r requirements.txt

##### Download movielens-1m dataset
    $ cd data/
    $ bash download_dataset.sh

## Train the LSPM model
    $ python3 main.py

## Citation
If this work is useful for your research, please cite our [paper](https://link.springer.com/chapter/10.1007/978-3-030-36808-1_26):
```
@inproceedings{chen2019lspm,
  title={LSPM: Joint Deep Modeling of Long-Term Preference and Short-Term Preference for Recommendation},
  author={Chen, Jie and Jiang, Lifen and Sun, Huazhi and Ma, Chunmei and Liu, Zekang and Zhao, Dake},
  booktitle={International Conference on Neural Information Processing},
  pages={237--246},
  year={2019},
  organization={Springer}
}
```
