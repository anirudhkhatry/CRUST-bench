from pathlib import Path
import os
import json
import multiprocessing
from openai import OpenAI

client = OpenAI()
FILE_PATH = Path(__file__)
CACHE_FILE = FILE_PATH.parent / "cache/gpt_cache.jsonl"
if not CACHE_FILE.parent.exists():
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
# Turn this off only when necessary
CACHE_FLAG = True
if not os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "w") as file:
        pass
with open(CACHE_FILE, "r", encoding="utf-8") as file:
    data = file.readlines()
CACHE = {json.loads(line)["prompt"]: json.loads(line) for line in data}


def get_text_from_openai_response(response):
    text = ""
    for output in response.output:
        if output.content is not None:
            for content in output.content:
                text += content.text
    return text

def call_gpt(messages, config, n=1):
    #NOTE from Sourya: Argument n is no longer used since the Responses API doesn't support it
    # but I have kept it around for function signature compatibility
    if config["model"] == "gpt-4o":
        response = client.responses.create(
            model=config["model"],
            input=messages,
            temperature=config["temperature"],
            max_output_tokens=config["max_tokens"],
            top_p=config["top_p"]
        )
    elif config["model"] == "o1" or config["model"] == "o1-mini":
        response = client.responses.create(
            model=config["model"],
            input=messages
        )
    else:
        raise ValueError("Model not supported")
    return response


def get_result(messages, lock, config):
    if config is None:
        raise ValueError("Config is None. Please look into the config.")
    current_prompt = (
        "\n".join([msg["content"] for msg in messages]) + f"\n{json.dumps(config)}"
    )

    if CACHE_FLAG and current_prompt in CACHE:
        return CACHE[current_prompt]

    try:
        response = call_gpt(messages, config)
    except Exception as e:
        return {"prompt": current_prompt, "response": str(e), "usage": {}}
    data = {
        "prompt": current_prompt,
        "response": get_text_from_openai_response(response),
        "usage": {
            "completion_tokens": response.usage.output_tokens,
            "prompt_tokens": response.usage.input_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
    if CACHE_FLAG:
        with lock:
            with open(CACHE_FILE, "a", encoding="utf-8") as file:
                file.write(json.dumps(data) + "\n")
            CACHE[current_prompt] = data
    return data


def get_result_n(messages, lock, config, n):
    if config is None or config["temperature"] == 0:
        raise ValueError("Config is None. Please look into the config.")
    current_prompt = (
        "\n".join([msg["content"] for msg in messages])
        + f"\n{json.dumps(config)}"
        + f"\n={n}"
    )

    if CACHE_FLAG and current_prompt in CACHE:
        return CACHE[current_prompt]

    try:
        print("Calling GPT")
        response = call_gpt(messages, config, n)
    except Exception as e:
        return {"prompt": current_prompt, "response": str(e), "usage": {}}
    data = {
        "prompt": current_prompt,
        "response": [get_text_from_openai_response(response)], #NOTE from Sourya: enclosing this in a list to preserve compatibility with what used to be here prior to using the Responses API
        "usage": {
            "completion_tokens": response.usage.output_tokens,
            "prompt_tokens": response.usage.input_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
    if CACHE_FLAG:
        with lock:
            with open(CACHE_FILE, "a", encoding="utf-8") as file:
                file.write(json.dumps(data) + "\n")
            CACHE[current_prompt] = data
    return data


def test_call_gpt(config):
    lock = multiprocessing.Lock()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in San Francisco?"},
    ]
    data = get_result(messages, lock, config)
    print(data)


if __name__ == "__main__":
    config1 = (FILE_PATH.parent / "configs/gpt_4o.json").read_text()
    config1 = json.loads(config1)
    test_call_gpt(config1)
