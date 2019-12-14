# Network-Anomaly-Detection
Final Project for CS221 - Detecting anomalous vertices within a large, directed graph dataset representing a network of users on Twitter

Final Report: https://github.com/anish-saha/Network-Anomaly-Detection/blob/master/CS_221_Project_Report.pdf

CodaLab Worksheet: https://worksheets.codalab.org/worksheets/0xb972380293b449d896f3c0ade600ee05

In this project, we try to construct a robust prediction algorithm to predict whether or not a social network user is a Sybil (fake account) using artificial intelligence. Our original dataset is a network dataset of over 5.3 million Twitter users.


## Models

Three models were used to perform node classification:
* A two-step link prediction/aggregation pipeline
![TwoStage](https://github.com/anish-saha/Network-Anomaly-Detection/blob/master/figures/Link_Prediction_Classifier.png)

* Graph Convolutional Network (GCN)
![GCN](https://github.com/anish-saha/Network-Anomaly-Detection/blob/master/figures/gcn.png)

* Graph Attention Networks (GAT)
![GAT](https://github.com/anish-saha/Network-Anomaly-Detection/blob/master/figures/gat.png)

## Setup
pip3 install --user -r requirements.txt

## Usage

### Two-stage Classifier
python3 eda.py

python3 sybil_detect.py

### GCN
cd pygcn/

python3 train.py

### GAT
cd pygat/

python3 train.py --sparse
