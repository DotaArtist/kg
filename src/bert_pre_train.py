# coding=utf-8


from bert_serving.client import BertClient


class BertPreTrain(object):
    def __init__(self, mode='remote'):
        if mode == 'remote':
            self.model = BertClient(ip='172.17.21.16', port=5555)
        else:
            self.model = BertClient(ip='127.0.0.1', port=5555)

    def get_output(self, sentence, _show_tokens=True):
        try:
            return self.model.encode(sentence, show_tokens=_show_tokens)
        except TypeError:
            print("sentence must be list！")


if __name__ == "__main__":
    from datetime import datetime
    aa = datetime.now()
    model = BertPreTrain()
    a = model.get_output(['输入内容文本最大长度128 ||| 什么 ', '输入内容文本最大长度128 ||| 什么 '], _show_tokens=False)
    bb = datetime.now()
    print((bb-aa).microseconds)
    print(a.shape[1])
