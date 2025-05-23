import os
from PIL import Image
from transformers import pipeline
from transformers import BitsAndBytesConfig
import torch
from dotenv import load_dotenv
load_dotenv()

model_variant = "4b-it"  # @param ["4b-it", "27b-text-it"]
model_id = f"google/medgemma-{model_variant}"

use_quantization = True  # @param {type: "boolean"}

model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if use_quantization:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
pipe = pipeline(
    "image-text-to-text",
    model=model_id,
    model_kwargs=model_kwargs,
)

pipe.model.generation_config.do_sample = False

def get_response(image_url,prompt,system_instruction="You are an expert radiologist and are looking at brain CT scans. You will receive a diagnosis prediction and an CT scan image and will help the user understand what is going on in the CT scan. Act as an explainer of the prediction and why that might be the case."):
    image_filename = os.path.basename(image_url)
    image = Image.open(image_filename)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]

    output = pipe(text=messages, max_new_tokens=400)
    response = output[0]["generated_text"][-1]["content"]
    return response

if __name__ == "__main__":
    get_response('.//epiduralh.png',"This scan is predicted to have an epidural hemorrhage.")


