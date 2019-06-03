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
            print(self.model.pos_tag(_sentence))
            # print(self.model.parse(_sentence))
            # print(self.model.dependency_parse(_sentence))

    def exit(self):
        self.model.close()


if __name__ == '__main__':
    model = SentenceParse()
    sent_1 = '良性反应性改变(轻度炎症)'
    model.parse(sentence=sent_1)
    model.exit()


# print(nlp.word_tokenize(sentence))
# print(nlp.pos_tag(sentence))
# print(nlp.ner(sentence))
# print(nlp.parse(sentence))
# print(nlp.dependency_parse(sentence))
