import time
from bert_base_skill_tag.client import BertClient

with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
    start_t = time.perf_counter()
    str = '职位描述：独立负责客户的项目需求沟通，整理项目简报，撰写策划方案；独立管理客户项目，把控项目执行进度，确保项目执行质量。负责与客户的日常沟通，了解并收集客户需求，并与公司各部门及供应商之间的沟通协调，保证项目良好执行。建立并保持与客户的密切沟通和良好关系，争取客户潜在和延展性需求的实现。任职要求：5年以上活动/展览公司从业经验独立负责过活动，发布会，展览及多媒体互动项目，有服务汽车行业客户经验者优先有较强的策略能力、创意思维和沟通能力优秀的文案策划能力和优秀的口头及书面表达能力细致认真、积极主动、性格开朗、讲求效率、乐于接受挑战、团队协作意识强富有责任心英文听说读写能力优秀者优先考虑。'
    rst = bc.encode([str])
    print('rst:', rst)
    print(time.perf_counter() - start_t)
