import json
import re

def has_citation(answer):
    return bool(re.search(r"\[Source\s+\d+\]", answer))

def basic_eval():
    with open("eval/test_questions.json", "r") as f:
        questions = json.load(f)

    print("RAG Evaluation Started")
    print("----------------------")

    for item in questions:
        print("Question:", item["question"])
        print("Expected citation:", item["must_have_citation"])

    print("----------------------")
    print("Manual evaluation file loaded successfully.")

if __name__ == "__main__":
    basic_eval()