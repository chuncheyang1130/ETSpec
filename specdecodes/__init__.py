"""specdecodes package.

Transformers 5 compatibility (Gemma 4 runs on transformers >= 5.5): since v5,
`apply_chat_template(..., tokenize=True)` returns a `BatchEncoding` by default
(`return_dict=True`), where v4 returned the token ids. About 25 call sites here
expect the ids (`input_ids.shape[1]`, `.to(device)`), so instead of touching each
one, a tokenizing call that does not ask for a dict gets `return_dict=False` --
exactly the v4 behaviour. On transformers 4.x this is a no-op (the default
already is False).
"""


def _restore_chat_template_ids() -> None:
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return
    original = PreTrainedTokenizerBase.apply_chat_template
    if getattr(original, "_specdecodes_ids_default", False):
        return

    def apply_chat_template(self, *args, **kwargs):
        if kwargs.get("tokenize", True) and "return_dict" not in kwargs:
            kwargs["return_dict"] = False
        return original(self, *args, **kwargs)

    apply_chat_template._specdecodes_ids_default = True
    apply_chat_template.__doc__ = original.__doc__
    PreTrainedTokenizerBase.apply_chat_template = apply_chat_template


_restore_chat_template_ids()
