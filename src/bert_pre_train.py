# coding=utf-8


from bert_serving.client import BertClient


class BertPreTrain(object):
    def __init__(self):
        self.model = BertClient(ip='172.17.21.16', port=5555)

    def get_output(self, sentence):
        try:
            return self.model.encode(sentence, show_tokens=True)
        except TypeError:
            print("sentence must be list！")


if __name__ == "__main__":
    from datetime import datetime
    aa = datetime.now()
    model = BertPreTrain()
    a = model.get_output(['输入内容文本最大长度128 ||| 什么 '])
    bb = datetime.now()
    print((bb-aa).microseconds)
    print(a)
