import base64
from pathlib import Path
from openai import OpenAI
client = OpenAI(
    api_key="APIKEY",
    base_url="https://api.featherless.ai/v1"
)
def encode_image_to_base64(image_path):
    """Encode a local image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def check_quest_complete(image_path,quest):
    base64_image = encode_image_to_base64(image_path)

    # Determine the image format for the data URL
    image_extension = Path(image_path).suffix.lower()
    if image_extension == '.png':
        data_url = f"data:image/png;base64,{base64_image}"
    elif image_extension in ['.jpg', '.jpeg']:
        data_url = f"data:image/jpeg;base64,{base64_image}"
    elif image_extension == '.webp':
        data_url = f"data:image/webp;base64,{base64_image}"
    else:
        data_url = f"data:image/jpeg;base64,{base64_image}"# Default to JPEG

    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The image uploaded is meant to be proof the user completed the following task: {quest}. Please return True if the evidence is sufficent and False otherwise. You should ONLY return the word true or false, nothing more"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content == "true"



def split_into_three(s):
    parts = s.split(",", 2)  # split on the first two commas only
    if len(parts) < 3:
        raise ValueError("Input string must contain at least two commas")
    return parts[0], parts[1], parts[2]
def resp(title,hp,mana):
    prompt = f"Write a short 2-3 sentence description in the style of a fantasy rpg game. Here is the information you know about them: they are a {title}, hp={hp}, mana={mana}. Don’t use a name or a gender. Make the description expand on their title, even if you have to make stuff up for them. This short description will be used on the bottom of a character card. Also write a more detailed title for them. Your response should read: “{title}, Title, Description” separated by commas. Do not put anything else."
    response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
    {"role": "user", "content": prompt}
    ]
    )
    return response.choices[0].message.content
#Example use (title, hp, mana)
#title, long_title, desc = split_into_three(resp("Mage", 35, 95))