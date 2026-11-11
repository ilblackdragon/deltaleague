"""Translate Qwen3.5 tokenizer XML tool calls into BFCL's JSON call notation."""
import json
import re

_TOOL = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
_FUNCTION = re.compile(r'<function=([\w.\-]+)>\s*(.*?)\s*</function>', re.DOTALL)
_PARAMETER = re.compile(r'<parameter=([\w.\-]+)>(.*?)</parameter>', re.DOTALL)


def normalize_tool_calls(text, functions):
    schemas={f['name']:f.get('parameters',{}).get('properties',{}) for f in functions}
    def replace(match):
        body=match.group(1)
        if not body.startswith('<function='):return match.group(0)
        function=_FUNCTION.fullmatch(body)
        if function is None:return match.group(0)
        name,parameters=function.groups();arguments={};position=0
        for parameter in _PARAMETER.finditer(parameters):
            key,value=parameter.groups()
            if parameters[position:parameter.start()].strip() or key in arguments:
                return match.group(0)
            value=value.strip();schema=schemas.get(name,{}).get(key,{})
            # Numeric-looking strings (IDs, ZIP codes) must remain strings.
            if schema.get('type')=='string':parsed=value
            else:
                try:parsed=json.loads(value)
                except (json.JSONDecodeError,ValueError):parsed=value
            arguments[key]=parsed;position=parameter.end()
        if parameters[position:].strip():return match.group(0)
        return '<tool_call>\n'+json.dumps({'name':name,'arguments':arguments})+'\n</tool_call>'
    return _TOOL.sub(replace,text)
