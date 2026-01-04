"""
Inference script for calendar event summarization.
"""

from transformers import pipeline

def main():
    summarizer = pipeline(
        "summarization",
        model="./calendar_summarizer",
        tokenizer="./calendar_summarizer",
    )

    test_events = [
        "Team standup meeting at 9am",
        "Lunch with Sarah",
        "Dentist appointment",
    ]

    for event in test_events:
        result = summarizer(
            "summarize: " + event,
            max_length=10,
            min_length=1,
        )
        print(f"{event} → {result[0]['summary_text']}")

if __name__ == "__main__":
    main()
