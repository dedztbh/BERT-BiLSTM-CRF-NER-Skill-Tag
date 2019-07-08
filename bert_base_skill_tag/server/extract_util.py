import re
import jieba.posseg as pseg

from .data import ndarray_to_tuple_array

with open('data/stopwords.txt', 'r', encoding='utf-8') as file:
    stop_words = file.read().split('\n')
stop_properties = []
stop_properties_general = list('efumoqy')


def desc_clean_clean(s):
    s = re.sub(r'[0-9一二三四五六七八九][.、]', ' ', s)
    s = re.sub(r'[-]{3,}|[*]{3,}', '', s)
    s = re.sub(r' ', '', s)
    return s


def filter_seg_result(pair):
    seg_word, properti = pair
    return (seg_word not in stop_words) and (properti not in stop_properties) \
           and (simplify_property(properti) not in stop_properties_general)


def simplify_property(p):
    if p == 'eng':
        return p

    return p[0]


def preprocess_input_with_properties(strs, split=False):
    def filter_words(x):
        seg_result = list(filter(filter_seg_result, pseg.cut(x)))
        return ''.join([w for w, p in seg_result])

    def clean_desc(s):
        s = desc_clean_clean(s)
        s = filter_words(s)
        if split:
            s = list(filter(lambda x: x != '', re.split(r'[。；！.;!]', s)))
        return s

    strs = [clean_desc(s) for s in strs]

    return strs


def preprocess_input_w_prop_embeddings(strs, return_tuple_array=False):
    def filter_words(x):
        seg_result = list(filter(filter_seg_result, pseg.cut(x)))
        x = ''.join([w for w, p in seg_result])

        seg_result = a2g(seg_result)
        prop_labels = []

        seg_word = ''
        properti = ''
        begin = True
        for ch in x:
            if seg_word == '':
                seg_word, properti = next(seg_result)
                # simplify
                properti = simplify_property(properti)
                begin = True
            if begin:
                mark_prefix = 'B-'
                begin = False
            else:
                mark_prefix = 'I-'
            mark = mark_prefix + properti.upper()
            prop_labels.append(mark)
            seg_word = seg_word[1:]

        assert len(x) == len(prop_labels)

        result = [list(x), prop_labels]

        return ndarray_to_tuple_array(result, 2) if return_tuple_array else result

    def clean_desc(s):
        s = desc_clean_clean(s)
        return filter_words(s)

    strs = [clean_desc(s) for s in strs]

    return strs


def a2g(x):
    return (n for n in x)
