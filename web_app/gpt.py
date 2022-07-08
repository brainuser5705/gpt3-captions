import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def send(prompt):
    response = openai.Completion.create(
        model='text-davinci-002',
        prompt=prompt,
        temperature=0.5,
    )

    return response.choices[0].text
