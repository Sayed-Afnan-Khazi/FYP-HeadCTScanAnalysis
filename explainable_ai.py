import ollama

def get_response(image_url,prompt,system_instruction="You are an expert radiologist and are looking at brain CT scans. You will receive a diagnosis prediction and an CT scan image and will help the user understand what is going on in the CT scan. Act as an explainer of the prediction and why that might be the case. Do not mention that you are a chatbot. Explain to the user in simple, easy to understand terms in 1-2 paragraphs. Don't ask any follow up questions."):
    res = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                'role': 'system',
                'content': system_instruction
            },
            {
                'role': 'user',
                'content': prompt,
                'images': [image_url]
            }
        ]
    )

    return res['message']['content']

if __name__ == "__main__":
    print(get_response('./epiduralh.png',"This scan is predicted to have an epidural hemorrhage."))


