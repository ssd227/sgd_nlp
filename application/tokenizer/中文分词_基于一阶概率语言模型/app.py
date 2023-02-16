# encoding=utf8

from sgd_nlp.core.tokenizer.max_probability_path_tokenizer import MaxProbabilityPathTokenizer
from sgd_nlp.core.common.read_file import *


def model_training(tokenizer, training_tokens_list):
    # step3: fit training data in model
    tokenizer.clear()
    tokenizer.train(training_tokens_list)

    # todo 保存统计数据，不要每次都从新训练
    # tokenizers.save_state()
    # tokenizers.load_state('')
    pass


def model_inference(tokenizer, test_data):
    for test_string in test_data:
        print(tokenizer.inference(test_string))


def load_data():
    # training_file path
    data_home = r"C:\Users\SGD\Desktop\sgd-代码库\sgd_deep_learning_framwork\sgd_nlp\data" \
                r"\tokenizer\backoff2005\training"
    train_file = "msr_training.utf8"
    training_file_path = os.path.join(data_home, train_file)

    # prepare training data in tokens form
    training_tokens_list = yield_tokens_list(training_file_path, encoding='utf-8')

    return training_tokens_list


def app():
    # step1 load data
    training_tokens_list = load_data()

    # step2 new model
    tokenizer = MaxProbabilityPathTokenizer()

    # step3 model train
    model_training(tokenizer, training_tokens_list)

    # step4 model inference
    test_data = [r'唯一打破这一岑寂的是这里的山泉。',
                 r'工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作',
                 r'多年来，中希贸易始终处于较低的水平，希腊几乎没有在中国投资。']
    model_inference(tokenizer, test_data)




if __name__ == '__main__':
    app()
