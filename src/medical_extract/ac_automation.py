# coding=utf-8


class Node(object):

    def __init__(self):
        self.next = {}  # 相当于指针，指向树节点的下一层节点
        self.fail = None  # 失配指针，这个是AC自动机的关键
        self.isWord = False  # 标记，用来判断是否是一个标签的结尾
        self.word = ""


class AcAutomation(object):

    def __init__(self):
        self.root = Node()

    def add_keyword(self, word):
        temp_root = self.root
        for char in word:
            if char not in temp_root.next:
                temp_root.next[char] = Node()
            temp_root = temp_root.next[char]
        temp_root.isWord = True
        temp_root.word = word

    def make_fail(self):
        temp_que = []
        temp_que.append(self.root)
        while len(temp_que) != 0:
            temp = temp_que.pop(0)
            p = None
            for key, value in temp.next.item():
                if temp == self.root:
                    temp.next[key].fail = self.root
                else:
                    p = temp.fail
                    while p is not None:
                        if key in p.next:
                            temp.next[key].fail = p.fail
                            break
                        p = p.fail
                    if p is None:
                        temp.next[key].fail = self.root
                temp_que.append(temp.next[key])

    def extract_keywords(self, content):
        p = self.root
        result = []
        current_position = 0

        while current_position < len(content):
            word = content[current_position]
            while word in p.next == False and p != self.root:
                p = p.fail

            if word in p.next:
                p = p.next[word]
            else:
                p = self.root

            if p.isWord:
                result.append(p.word)
            else:
                current_position += 1
        return result


def test():
    ac = AcAutomation()
    ac.add_keyword('头痛')
    ac.add_keyword('头晕')
    ac.add_keyword('发热')

    print(ac.extract_keywords('头痛头晕'))


if __name__ == '__main__':
    test()
