import json
from delta_cli.public_eval.bfcl_tool_calls import normalize_tool_calls as normalize

def test_qwen_xml_arguments_preserve_schema_types():
    text='<tool_call>\n<function=lookup>\n<parameter=id>\n00123\n</parameter>\n<parameter=count>\n10\n</parameter>\n<parameter=filters>\n{"active":true}\n</parameter>\n</function>\n</tool_call>'
    functions=[{'name':'lookup','parameters':{'properties':{'id':{'type':'string'},'count':{'type':'integer'},'filters':{'type':'dict'}}}}]
    result=json.loads(normalize(text,functions).removeprefix('<tool_call>\n').removesuffix('\n</tool_call>'))
    assert result=={'name':'lookup','arguments':{'id':'00123','count':10,'filters':{'active':True}}}


def test_legacy_json_and_malformed_calls_are_not_fabricated():
    for text in ['ordinary answer','<tool_call>\n{"name":"f","arguments":{}}\n</tool_call>',
                 '<tool_call><function=f><parameter=x>1</function></tool_call>',
                 '<tool_call><function=f><parameter=x>1</parameter><parameter=x>2</parameter></function></tool_call>']:
        assert normalize(text,[])==text


def test_parallel_xml_calls_and_multiline_strings():
    text='<tool_call><function=a><parameter=text>hello\nworld</parameter></function></tool_call>\n<tool_call><function=b></function></tool_call>'
    out=normalize(text,[{'name':'a','parameters':{'properties':{'text':{'type':'string'}}}}])
    assert out.count('<tool_call>')==2 and 'hello\\nworld' in out and '"name": "b"' in out
