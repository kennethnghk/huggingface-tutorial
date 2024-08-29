## https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt
from pprint import pprint
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")

pprint(result)

generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to")

pprint(result)

classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business", "movie"],
)

pprint(result)