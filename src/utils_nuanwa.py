import logging
import argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    body = get_body_entity(tag_seq, char_seq)
    symp = get_symp_entity(tag_seq, char_seq)
    dise = get_dise_entity(tag_seq, char_seq)
    cure = get_cure_entity(tag_seq, char_seq)
    check = get_check_entity(tag_seq, char_seq)
    return body, symp, dise, cure, check


def get_body_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-body':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i + 1 == length:
                LOC.append(loc)
        if tag in ['I-body', 'E-body']:
            if 'loc' not in locals().keys():
                loc = char
            else:
                loc += char
            if i + 1 == length:
                LOC.append(loc)
        if tag not in ['B-body', 'I-body', 'E-body']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_symp_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-symp':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i + 1 == length:
                LOC.append(loc)
        if tag in ['I-symp', 'E-symp']:
            if 'loc' not in locals().keys():
                loc = char
            else:
                loc += char
            if i + 1 == length:
                LOC.append(loc)
        if tag not in ['B-symp', 'I-symp', 'E-symp']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_check_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-chec':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i + 1 == length:
                LOC.append(loc)
        if tag in ['I-chec', 'E-chec']:
            if 'loc' not in locals().keys():
                loc = char
            else:
                loc += char
            if i + 1 == length:
                LOC.append(loc)
        if tag not in ['B-chec', 'I-chec', 'E-chec']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_dise_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-dise':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i + 1 == length:
                LOC.append(loc)
        if tag in ['I-dise', 'E-dise']:
            if 'loc' not in locals().keys():
                loc = char
            else:
                loc += char
            if i + 1 == length:
                LOC.append(loc)
        if tag not in ['B-dise', 'I-dise', 'E-dise']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_cure_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-cure':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i + 1 == length:
                LOC.append(loc)
        if tag in ['I-cure', 'E-cure']:
            if 'loc' not in locals().keys():
                loc = char
            else:
                loc += char
            if i + 1 == length:
                LOC.append(loc)
        if tag not in ['B-cure', 'I-cure', 'E-cure']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
