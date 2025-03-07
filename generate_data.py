from datasets import load_dataset, Dataset
import random
import concurrent.futures
import json
from tqdm import tqdm

from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.identity import DefaultAzureCredential

#DEFAULT_AZURE_ENDPOINT = "https://lifangopenai.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
#DEFAULT_AZURE_ENDPOINT = "https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.en"

dataset = load_dataset("wikipedia", "20220301.en", cache_dir="./hf_cache", trust_remote_code=True)
train_dataset = dataset["train"]
print("Total training recoreds", len(train_dataset))

train_dataset = list(train_dataset)
sampled_items = random.sample(train_dataset, 1000)
text_samples = [item["text"] for item in sampled_items]

#endpoint = "https://lifangopenai.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
endpoint = "https://lifangopenai.openai.azure.com/"
model_name = "gpt-4o"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=DefaultAzureCredential(),
)

print("Client Created")

def generate_variation_prompt(text):
    num_cards = random.randint(3,20)
    prompt_style = [
        "Create {} flashcards for the following text:",
        "Produce {} engaging flashcards based on the text below:",
        "From the following artical, generate {} informative flashcards:" 
    ]

    prompt_intro = random.choice(prompt_style).format(num_cards)

    flashcards_formats = [
        """
[
    {
    "front":"",
    "back":""
    }
]
""",
"""
[
    {
    "question:"",
    "answer":""
    }
]
"""
    ]
    flashcards_format = random.choice(flashcards_formats)
    full_prompt = f"""
{prompt_intro}
<text>
{text}
</text>
Generate the flashcards in the following JSON format:
<response_format>
{flashcards_format}
</respons_format>
"""
    return full_prompt

def process_sample(text):
    prompt = generate_variation_prompt(text)
    try:
        response = client.chat.completions.create(
            model = "gpt-4o",
            messages= [{"role" : "user", "content" : prompt}],
            max_tokens=1000,
            temperature=0.75
        )
        message = response.choices[0].message.content
        if message is None:
            return {
                "original_text" : text,
                "prompt" : prompt,
                "flashcards" : message
            }
    except Exception as e:
        return None
    
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_text = {executor.submit(process_sample, text) : text for text in text_samples}
        for future in tqdm(concurrent.futures.as_completed(future_to_text), total=len(future_to_text), desc="Processing samples"):
            result = future.result()
            if result is not None:
                results.append(result)
                print("got result", len(result))
    
    flashcards_dataset = Dataset.from_list(results)
    flashcards_dataset.save_to_disk("flashcards_dataset")

    print("Saved generated flashcards dataset to disk as 'flashcards_dataset'.")
    for idx, entry in enumerate(results):
        print(f"--- Entry {idx + 1} ---")
        print("prompt:")
        print(entry["prompty"])
        print("fFlashcards:")
        print(entry["flashcards"])
