from vllm import LLM, SamplingParams
from PIL import Image

model_name = "Allen8/TVC-72B"
llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=8,
    )

question = "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Subtract all red things. Subtract all tiny matte balls. How many objects are left?\nPlease answer the question using a long-chain reasoning style and think step by step."
placeholder = "<|image_pad|>"
prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
f"{question}<|im_end|>\n"
"<|im_start|>assistant\n")

sampling_params = SamplingParams(
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    stop_token_ids=[],
    repetition_penalty=1.05,
    max_tokens=8192
)

image = Image.open("../images/case1.png")
inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        }

outputs = llm.generate([inputs], sampling_params=sampling_params)
print(outputs[0].outputs[0].text)