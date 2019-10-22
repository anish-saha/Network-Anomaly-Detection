import os

from avd.utils.label_encoder import BinaryLabelEncoder

label_encoder = BinaryLabelEncoder()
graph_max_edge_number = 10000000
save_progress_interval = 200000

TWITTER_URL = "http://proj.ise.bgu.ac.il/sns/datasets/twitter.csv.gz"
TWITTER_LABELS_URL = "http://proj.ise.bgu.ac.il/sns/datasets/twitter_fake_ids.csv"

DATA_DIR_NAME = "data"

DATA_DIR = os.path.expanduser(os.path.join(os.getcwd(), DATA_DIR_NAME))
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

TEMP_DIR = os.path.expanduser(os.path.join(DATA_DIR, 'temp'))
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)

OUTPUT_DIR = os.path.expanduser(os.path.join(DATA_DIR, 'output'))
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
