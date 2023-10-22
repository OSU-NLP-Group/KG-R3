# Generate retriever subgraphs from scratch


## Minerva Retriever

1. Train the MINERVA model using code from the repo https://github.com/shehzaadzd/MINERVA provided by [MINERVA](https://arxiv.org/pdf/1711.05851.pdf) authors.

2: Dump the train, valid, and test paths in the directories `MINERVA_MODEL_DIR/train_beam`, `MINERVA_MODEL_DIR/dev_beam`, and `MINERVA_MODEL_DIR/test_beam` respectively.

3. Convert the paths into a LMDB database

### FB15K-237
```
cd utils/
# FW direction

python dump_minerva_subgraphs.py --dataset-path ../data/FB15K-237/ --path-dir MINERVA_MODEL_DIR/train_beam --db-path ../data/FB15K-237/subgraphs_minerva_train --split train
python dump_minerva_subgraphs.py --dataset-path ../data/FB15K-237/ --path-dir MINERVA_MODEL_DIR/dev_beam --db-path ../data/FB15K-237/subgraphs_minerva_valid --split valid
python dump_minerva_subgraphs.py --dataset-path ../data/FB15K-237/ --path-dir MINERVA_MODEL_DIR/test_beam --db-path ../data/FB15K-237/subgraphs_minerva_test --split test

# BW direction

python dump_minerva_subgraphs.py --dataset-path ../data/FB15K-237/ --path-dir MINERVA_MODEL_DIR/train_beam --db-path ../data/FB15K-237/subgraphs_minerva_train_rev --split train --reverse
python dump_minerva_subgraphs.py --dataset-path ../data/FB15K-237/ --path-dir MINERVA_MODEL_DIR/dev_beam --db-path ../data/FB15K-237/subgraphs_minerva_valid_rev --split valid --reverse
python dump_minerva_subgraphs.py --dataset-path ../data/FB15K-237/ --path-dir MINERVA_MODEL_DIR/test_beam --db-path ../data/FB15K-237/subgraphs_minerva_test_rev --split test --reverse
```

### WN18RR

```
# FW direction
python dump_minerva_subgraphs_segment.py --dataset-path ../data/WN18RR/ --path-dir MINERVA_MODEL_DIR/test_beam --db-path ../data/WN18RR/subgraphs_minerva_beam_40_seg_test --split test
python dump_minerva_subgraphs_segment.py --dataset-path ../data/WN18RR/ --path-dir MINERVA_MODEL_DIR/train_beam --db-path ../data/WN18RR/subgraphs_minerva_beam_40_seg_train --split train
python dump_minerva_subgraphs_segment.py --dataset-path ../data/WN18RR/ --path-dir MINERVA_MODEL_DIR/dev_beam --db-path ../data/WN18RR/subgraphs_minerva_beam_40_seg_valid --split valid

# BW direction
python dump_minerva_subgraphs_segment.py --dataset-path ../data/WN18RR/ --path-dir MINERVA_MODEL_DIR/test_beam --db-path ../data/WN18RR/subgraphs_minerva_beam_40_seg_test_rev --split test --reverse
python dump_minerva_subgraphs_segment.py --dataset-path ../data/WN18RR/ --path-dir MINERVA_MODEL_DIR/train_beam --db-path ../data/WN18RR/subgraphs_minerva_beam_40_seg_train_rev --split train --reverse
python dump_minerva_subgraphs_segment.py --dataset-path ../data/WN18RR/ --path-dir MINERVA_MODEL_DIR/dev_beam --db-path ../data/WN18RR/subgraphs_minerva_beam_40_seg_valid_rev --split valid --reverse
```

## BFS Retriever
If the BFS subgraphs corresponding to the given hyperparameters are not found in the path, the code will auto-generate them while running the main training script (method `links2subgraphs()` in dataset.py)

## One-hop Retriever
If the one-hop subgraphs corresponding to the given hyperparameters are not found in the path, the code will auto-generate them while running the main training script (method `links2subgraphs()` in dataset.py)

