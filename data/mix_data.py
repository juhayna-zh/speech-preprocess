from make_data import make_dataset

#TODO: 改为自己的配置
DATASET_SIZE = 80000 #数据量
SAVE_DATASET_DIR = '/mnt/lustre/sjtu/home/hnz01/code/AILabInTask1/speech-preprocess/data/DatasetTrain80k' #待生成数据集的保存位置
INDEX_JSONL_PATH = '/mnt/lustre/sjtu/home/hnz01/code/AILabInTask1/speech-preprocess/data/train_index.jsonl' #数据索引文件的位置
SPEECH_DATA_DIR = '/mnt/lustre/sjtu/home/hnz01/data/AISHELL-3/train/wav' #原始语音数据的目录
NOISE_DATA_DIR = '/mnt/lustre/sjtu/home/hnz01/data/NoiseX-92' #原始噪声数据的目录

make_dataset(DATASET_SIZE, 
             save_dir=SAVE_DATASET_DIR,
             index_jsonl_path=INDEX_JSONL_PATH,
             parent_dir=SPEECH_DATA_DIR,
             noise_dir=NOISE_DATA_DIR)