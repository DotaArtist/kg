# coding=utf-8
import re
from stanfordcorenlp import StanfordCoreNLP


class SentenceParse(object):
    def __init__(self, path='../data/stanford-corenlp-full-2018-10-05/', lang='zh'):
        self.path = path
        self.lang = lang
        self.sentence = None
        self.model = StanfordCoreNLP(path_or_host=self.path, lang=self.lang)

    def parse(self, sentence):
        self.sentence = sentence
        for _sentence in re.split('[。.\n]', self.sentence.strip('。.\n')):
            print("=========")
            print(_sentence)
            # print(self.model.pos_tag(_sentence))
            _ = self.model.parse(_sentence)  # 依存句法分析 DP
            print(_.replace('\t', '').replace('\r\n', ''))
            print(type(_))
            # print(self.model.dependency_parse(_sentence))  # 语义依存关系 SDP

    def exit(self):
        self.model.close()


if __name__ == '__main__':
    model = SentenceParse()
    sent_1 = '乳酸脱氢酶测定试剂盒ldh速率法'
    # sent_1 = '中华人民共和国(中国)很强大'
    model.parse(sentence=sent_1)
    model.exit()


# print(nlp.word_tokenize(sentence))
# print(nlp.pos_tag(sentence))
# print(nlp.ner(sentence))
# print(nlp.parse(sentence))
# print(nlp.dependency_parse(sentence))
