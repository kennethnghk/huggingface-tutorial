## https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt
from pprint import pprint
from transformers import pipeline

print('########################################')
print('######## sentiment-analysis by default model')
classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
pprint(result)

print('########################################')
print('######## text-generation by default model')
generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to")
pprint(result)

print('########################################')
print('######## zero-shot-classification by default model')
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business", "movie"],
)
pprint(result)

print('########################################')
print('######## text-generation by distilgpt2')
generator = pipeline("text-generation", model="distilgpt2")
result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
pprint(result)

## https://huggingface.co/openai-community/gpt2
print('########################################')
print('######## text-generation by GPT-2')
generator = pipeline("text-generation", model="gpt2")
result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
pprint(result)