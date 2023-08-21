from make_data import make_audio_index

#TODO: 改为自己设备上的路径
SPEECH_DATA_DIR= '/mnt/lustre/sjtu/home/hnz01/data/AISHELL-3/train/wav' #原始语音数据的目录
INDEX_JSONL_PATH = '/mnt/lustre/sjtu/home/hnz01/code/AILabInTask1/speech-preprocess/data/train_index.jsonl' #待生成的数据索引文件的保存位置

if __name__ == '__main__':
    make_audio_index(SPEECH_DATA_DIR, INDEX_JSONL_PATH)
