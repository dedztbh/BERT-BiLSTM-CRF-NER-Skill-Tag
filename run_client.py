# -!- coding: utf-8 -!-

import time
import json
from bert_base_skill_tag.client import BertClient
from extract_util import preprocess_input_w_prop_embeddings


def get_LBL_entity(tag_seq, char_seq):
    length = len(char_seq)
    LBL = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LBL':
            if 'lbl' in locals().keys():
                LBL.append(lbl)
                del lbl
            lbl = char
            if i + 1 == length:
                LBL.append(lbl)
        if tag == 'I-LBL':
            lbl += char
            if i + 1 == length:
                LBL.append(lbl)
        if tag not in ['I-LBL', 'B-LBL']:
            if 'lbl' in locals().keys():
                LBL.append(lbl)
                del lbl
            continue
    return LBL


with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
    start_t = time.perf_counter()
    str = """职位描述： 1.软件测试工程师主要职责是测试公司自主研发的各种软件产品，提供详细的测试结果； 2.能够学习相关的主流技术，结合产品进行测试； 3.能够根据客户需求，完成测试用例的编写和执行，同时能够按照测试计划，完成测试任务。 职位要求： 1.本科及以上学历，计算机或软件专业，对黑盒测试感兴趣，愿意从事测试工作； 2.逻辑清晰，表达清楚；有较强的责任心，抗压能力和团队合作精神，快速学习和适应环境能力； 3.具备一定的计算机基础知识，对操作系统和数据库有基础的了解和认识； 4英语四级以上，能够阅读英文文档，英语或日语熟练表达使用者优先； 5.对有编程经验，了解VM，SharePoint，Online Service，SQL Server，或者有过相关工作经验者优先。"""
    str = preprocess_input_w_prop_embeddings([str], return_tuple_array=True, split=True)[0]
    jsons = [json.dumps({'input': sp[0], 'prop': sp[1]}, ensure_ascii=False) for sp in str]
    # print(jsons)
    rst = bc.encode(jsons)
    # # print('rst:', rst)
    finalset = set()
    for i, r in enumerate(rst):
        finalset = set(get_LBL_entity(r, str[i][0])) | finalset
    print(finalset)
    print(time.perf_counter() - start_t)
