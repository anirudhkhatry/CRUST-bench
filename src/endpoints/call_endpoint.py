import endpoints.claude as claude
import endpoints.claude37 as claude37
import endpoints.gpt as gpt
import endpoints.qwq as qwq
import endpoints.huggingface as hf
# uncomment the next line if you want to use gemini
# import endpoints.gemini as gemini
import multiprocessing
import json
import sys
import endpoints.vllm_client as vclient


def get_result(messages, lock, model, config):
    if model == "claude":
        return claude.get_result(messages, lock, config)
    elif model == "claude37":
        return claude37.get_result(messages, lock, config)
    elif model in ["gpt-4o", "o1-mini", "o1", "gpt-oss-120b"]:
        return gpt.get_result(messages, lock, config)
    elif (
        model == "QwQ-32B-Preview"
        or model == "Qwen2.5-Coder-32B"
        or model == "Virtuoso-Medium-v2"
        or model == "Meta-Llama-3-8B"
    ):
        m = vclient.VLLMServer(model)
        return m.get_result(messages, lock, config)
    elif model == "gemini":
        return gemini.get_result(messages, lock, config)
    elif model == 'gpt-oss-20b':
        return hf.get_result_gptoss(messages, config)
    else:
        raise ValueError("Invalid model")


def get_result_n(messages, lock, model, config, n):
    if model == "gpt-4o" or model == "o1-mini" or model == "o1":
        return gpt.get_result_n(messages, lock, config, n)
    elif model == "claude":
        return claude.get_result_n(messages, lock, config, n)
    else:
        raise ValueError("Invalid model")


if __name__ == "__main__":
    print(
        get_result(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather in San Francisco?"},
            ],
            multiprocessing.Lock(),
            "claude",
            None,
        )
    )
    print(
        get_result(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather in San Francisco?"},
            ],
            multiprocessing.Lock(),
            "gpt",
            None,
        )
    )
