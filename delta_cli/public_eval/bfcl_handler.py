"""Use the exact checkpoint tokenizer with BFCL's Qwen tool-call parser."""
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from overrides import override
import threading
from .bfcl_tool_calls import normalize_tool_calls
from .bfcl_errors import ContextExhausted


class Qwen35Handler(QwenFCHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A handler is shared by the inference pool; schemas belong to each request.
        self._tool_schemas = threading.local()

    @override
    def _format_prompt(self, messages, function):
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
            tools=[{'type': 'function', 'function': f} for f in function] or None,
            add_generation_prompt=True, enable_thinking=getattr(self, 'validation_generation', {}).get('enable_thinking', False))

    @override
    def _query_prompting(self, inference_data: dict):
        self._tool_schemas.functions = inference_data['function']
        generation = getattr(self, 'validation_generation', {})
        if not generation:
            return super()._query_prompting(inference_data)
        import time
        from .generation import sampling
        prompt = self._format_prompt(inference_data['message'], inference_data['function'])
        inference_data['inference_input_log'] = {'formatted_prompt': prompt}
        prompt_tokens=len(self.tokenizer.tokenize(prompt))
        available = self.max_context_length - prompt_tokens - 2
        if available < 1:
            raise ContextExhausted(prompt_tokens,self.max_context_length)
        options = sampling(generation)
        extra = {key: options.pop(key) for key in list(options) if key not in ('temperature', 'top_p', 'presence_penalty')}
        for key in ('stop_token_ids', 'skip_special_tokens'):
            if hasattr(self, key):
                extra[key] = getattr(self, key)
        started = time.monotonic()
        response = self.client.completions.create(model=self.model_path_or_id, prompt=prompt,
            max_tokens=min(generation['max_tokens'], available), seed=generation['seed'],
            **options, extra_body=extra, timeout=1800)
        return response, time.monotonic() - started

    @override
    def _parse_query_response_prompting(self, api_response) -> dict:
        api_response = api_response.model_copy(deep=True)
        if getattr(self, 'validation_generation', {}).get('enable_thinking'):
            from .generation import final_answer
            # Tool calls in reasoning must never be executed or credited.
            api_response.choices[0].text = final_answer(api_response.choices[0].text or '')
        api_response.choices[0].text = normalize_tool_calls(
            api_response.choices[0].text or '', getattr(self._tool_schemas, 'functions', []))
        return super()._parse_query_response_prompting(api_response)
