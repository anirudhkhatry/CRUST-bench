import json
import re
import transformers
from typing import Any, Optional, Union


def create_pipeline(model_id: str) -> transformers.pipeline:
    """
    Create and return a text generation pipeline for the given `model_id`.
    dtype and device_map are automatically assigned, these will be set to the best possible **if the accelerate library is insalled.**
    """
    pipe = transformers.pipeline(
        "text-generation",
        model = model_id,
        dtype = "auto",
        device_map = "auto"
    )
    return pipe


def run_pipeline(
    pipe: transformers.pipeline,
    prompt: Union[str, list[dict[str,str]]],
    max_new_tokens: int,
    gen_kwargs: dict = {},
    reasoning: Optional[str] = None
) -> str:
    """
    Return the results of running the given `prompt` through pipeline `pipe` to generate `max_new_tokens` of text under `gen_kwargs` generation settings.

    Example 1:
    ```
    run_pipeline(
        pipe,
        prompt = [ #NOTE: this style of prompt is required for OpenAI gpt-oss
            {"role": "system", "content": "Reasoning: medium"},
            {"role": "user", "content": "Explain quantum mechanics clearly and concisely."}
        ],
        max_new_tokens = 256, 
        gen_kwargs = {
            'temperature': 1.0,
            'top_p': 1.0
        }
    )
    ```

    Example 2:
    ```
    run_pipeline(
        pipe,
        prompt = "Hello, how are you?",
        max_new_tokens = 20
    )
    ```

    A few notes on runtime:
    - For gpt_oss, reasoning doesn't seem to have any effect on runtime. But CoT length reduces for lower reasoning.
    - Runtime increases proportionate to max_new_tokens.
    """
    reasoning_instruction = f'Reasoning: {reasoning}'
    if type(prompt) == str:
        prompt += f'\n{reasoning_instruction}'
    else:
        for p in prompt:
            if p['role'] == 'system':
                p['content'] += f'\n{reasoning_instruction}'
    output = pipe(
        prompt,
        return_full_text = False,
        max_new_tokens = max_new_tokens,
        **gen_kwargs
    )
    return output[0]["generated_text"]


def postprocess_pipeline_output_gptoss(pipe_out: str) -> tuple[str,str]:
    """
    Bespoke function to postprocess pipeline returned text for GPT-OSS and parse into output and chain-of-thought segments.

    Returns:
    - out: Model actual output text.
    - cot: Model chain-of-thought text.
    """
    m = re.match(r"analysis(?P<cot>(.|\n)*)assistantfinal(?P<out>(.|\n)*)\Z", pipe_out)
    assert m is not None, f"Pipeline output not properly formatted. Pipeline output:\n{pipe_out}"
    return m.group('out'), m.group('cot')


def get_result_gptoss(messages: list[dict[str,str]], config: dict[str,Any]) -> dict[str,Any]:
    current_prompt = (
        "\n".join([msg["content"] for msg in messages]) + f"\n{json.dumps(config)}"
    )
    try:
        pipe = create_pipeline(model_id = f"openai/{config['model']}")
        pipe_out = run_pipeline(
            pipe = pipe,
            prompt = messages,
            max_new_tokens = config['max_new_tokens'],
            gen_kwargs = config['gen_kwargs'],
            reasoning = 'minimal'
        )
        out, cot = postprocess_pipeline_output_gptoss(pipe_out = pipe_out)
    except Exception as e:
        return {
            "prompt": current_prompt,
            "response": str(e),
            "cot": "",
            "usage": {}
        }
    return {
        "prompt": current_prompt,
        "response": out,
        "chain_of_thought": cot,
        "usage": {}
    }
