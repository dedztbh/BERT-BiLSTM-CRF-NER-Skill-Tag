import re
import jieba.posseg as pseg
import string

from data import ndarray_to_tuple_array, tuple_array_to_ndarray

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


def is_letter(c):
    return c in string.ascii_letters


def preprocess_input_w_prop_embeddings_e(strs, return_tuple_array=False):
    eng_list = []

    def filter_words(x):
        nonlocal eng_list
        seg_result = list(filter(filter_seg_result, pseg.cut(x)))
        x = ''.join([w for w, p in seg_result])

        seg_result = a2g(seg_result)
        prop_labels = []

        new_x = ''

        seg_word = ''
        properti = ''
        begin = True
        reading_english = ''
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

            seg_word = seg_word[1:]

            if is_letter(ch):
                if not len(reading_english):
                    new_x += 'e'
                    prop_labels.append(mark)
                reading_english += ch
            else:
                if len(reading_english):
                    eng_list.append(reading_english)
                    reading_english = ''
                new_x += ch
                prop_labels.append(mark)

        if len(reading_english):
            eng_list += reading_english

        x = new_x
        assert len(x) == len(prop_labels)

        result = [x, prop_labels]

        return result

    def clean_desc(s):
        nonlocal eng_list
        s = desc_clean_clean(s)
        s = filter_words(s)

        # split
        s0s = re.split(r'[。；！.;!]', s[0])
        s1s = []
        i = 0
        for s0 in s0s:
            seg_len = len(s0)
            s1s.append(s[1][i:i + seg_len])
            i += seg_len + 1
        for s0, s1 in zip(s0s, s1s):
            assert len(s0) == len(s1)
        s = tuple_array_to_ndarray(list(filter(lambda x: x[0] != '', zip(s0s, s1s))))

        new_s0 = []
        for sent in s[0]:
            new_s0_single = []
            if 'e' in sent:
                for i, frag in enumerate(sent.split('e')):
                    if i:
                        new_s0_single.append(eng_list[0])
                        new_s0_single += list(frag)
                        eng_list = eng_list[1:]
                    else:
                        new_s0_single += list(frag)
            else:
                new_s0_single += sent
            new_s0.append(new_s0_single)
        assert len(eng_list) == 0
        s[0] = new_s0

        if return_tuple_array:
            s = ndarray_to_tuple_array(s)
        return s

    strs = [clean_desc(s) for s in strs]

    return strs


def a2g(x):
    return (n for n in x)
