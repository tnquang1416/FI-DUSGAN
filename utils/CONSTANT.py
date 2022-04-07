'''
Created on Nov 21, 2020

@author: Quang Tran
'''
import torch.cuda

MESS_CONFLICT_PROGRESS = "/!\Progresses of networks are different."
MESS_CONFLICT_DATA_SIZES = "/!\Different in size of output and ground truth."
MESS_NO_CUDA = "/!\Cuda is not available."
MESS_NO_VERSION = "!!!No defined version."
MESS_CONFLICT_SHAPE = "!!!Inappropriate shape."
MESS_NO_PATCH_SIZE = "!!!Unsupported size of patch."
MESS_NO_VERSION = "No specific version"

TEST_PATH = "tests"
TEST_PATH_DEBUG = "tests_fasttrain"
LOG_FILE = "log.txt"
TRAINING_OUTPUT = "E:/2020_fi_training/train_"
TRAINING_OUTPUT_DEBUG = "E:/2020_fi_training_fast/train_"

# saved model's label
LABEL_EPOCH = "epoch"
LABEL_MODEL_STATE_DICT = "state_dict"
LABEL_OPTIMIZER_STATE_DICT = "optimizer"
LABEL_LOSS = "loss"
LABEL_VERSION = "version"

# dataset paths
D_UCF101_BODY_TRAIN_DEV = 'D:/gan_testing/data/ucf101_body_motion/train_dev'
D_UCF101_BODY_TRAIN_DIR_DEV_5 = 'D:/gan_testing/fi_dataset_dirs/5/db_body_train_dev_128'
D_UCF101_BODY_TRAIN_DIR_DEV_3 = 'D:/gan_testing/fi_dataset_dirs/3/db_body_train_dev_128'
D_UCF101_BODY_TEST_DEV = 'D:/gan_testing/data/ucf101_body_motion/test_dev'
D_UCF101_BODY_TEST_DIR_DEV_5 = 'D:/gan_testing/data/ucf101_body_motion/test_dev'
D_UCF101_TRAIN_DIR_RELEASE_5 = 'D:/gan_testing/fi_dataset_dirs/5/db_body_train_release_128'
D_UCF101_TRAIN_DIR_RELEASE_3 = 'D:/gan_testing/fi_dataset_dirs/3/db_body_train_release_128'
D_UCF101_VAL_DIR_RELEASE_3 = 'D:/gan_testing/fi_dataset_dirs/3/db_body_train_release_128_small'
D_UCF101_ALICE_DEV = 'D:/pyProject/GANs/data/custom/alice'

D_VIMEO_TRAIN_PATHS_3 = 'D:/gan_testing/fi_dataset_dirs/3/vimeo_training_release/tri_vallist.txt'
D_VIMEO_TEST_PATHS_3 = 'E:/dataset/vimeo_triplet/vimeo_triplet/tri_testlist.txt'
D_VIMEO_VAL_PATHS_3 = 'D:/gan_testing/fi_dataset_dirs/3/vimeo_training_release/tri_vallist.txt'
D_VIMEO_COLLECTION = [D_VIMEO_TRAIN_PATHS_3, D_VIMEO_TEST_PATHS_3, D_VIMEO_VAL_PATHS_3]

D_UCF101_BODY_TRAIN = 'D:/gan_testing/data/ucf101_body_motion/train'

Tensor = torch.cuda.FloatTensor  # @UndefinedVariable

# version
BASELINE = "baseline"

VERSION_0_0 = "0.0."
VERSION_0_01 = VERSION_0_0 + "1"
VERSION_0_02 = VERSION_0_0 + "2"

VERSION_1_0 = "1.0."
VERSION_1_00 = VERSION_1_0 + "0"
VERSION_1_01 = VERSION_1_0 + "1"
VERSION_1_02 = VERSION_1_0 + "2"
VERSION_1_04 = VERSION_1_0 + "4"
VERSION_1_05 = VERSION_1_0 + "5"
VERSION_1_06 = VERSION_1_0 + "6"
VERSION_1_07 = VERSION_1_0 + "7"
VERSION_1_08 = VERSION_1_0 + "8"

VERSION_1_1 = "1.1."
VERSION_1_10 = VERSION_1_1 + "0"
VERSION_1_11 = VERSION_1_1 + "1"
VERSION_1_12 = VERSION_1_1 + "2"
VERSION_1_13 = VERSION_1_1 + "3"

VERSION_1_2 = "1.2."
VERSION_1_20 = VERSION_1_2 + "0"
VERSION_1_21 = VERSION_1_2 + "1"
VERSION_1_22 = VERSION_1_2 + "2"
VERSION_1_23 = VERSION_1_2 + "3"
VERSION_1_24 = VERSION_1_2 + "4"
VERSION_1_25 = VERSION_1_2 + "5"
VERSION_1_26 = VERSION_1_2 + "6"
VERSION_1_27 = VERSION_1_2 + "7"
VERSION_1_28 = VERSION_1_2 + "8"
VERSION_1_29 = VERSION_1_2 + "9"

VERSION_1_3 = "1.3."
VERSION_1_30 = VERSION_1_3 + "0"

VERSION_2_0 = "2.0."
VERSION_2_00 = VERSION_2_0 + "0"
VERSION_2_01 = VERSION_2_0 + "1"
VERSION_2_02 = VERSION_2_0 + "2"

VERSION_LIST = [BASELINE,
                VERSION_1_20, VERSION_1_24, VERSION_1_25,
                VERSION_2_00, VERSION_2_01, VERSION_2_02]
