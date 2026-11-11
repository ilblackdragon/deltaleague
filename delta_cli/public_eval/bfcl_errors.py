"""Model-limit outcomes, distinct from infrastructure or evaluator failures."""


class ContextExhausted(ValueError):
    def __init__(self, prompt_tokens, context_limit):
        self.prompt_tokens=prompt_tokens
        self.context_limit=context_limit
        super().__init__('BFCL context exhausted')
