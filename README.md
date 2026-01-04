# Calendar ML Bot

This project trains a machine learning model to turn calendar event text into short summaries.

**Example**
- `"Team standup meeting at 9am"` → `"Standup"`
- `"Lunch with Sarah"` → `"Lunch"`

---

## What this project does

- Takes calendar event text as input
- Uses a pre-trained **T5-small** model
- Fine-tunes it to produce short summaries or labels

This was built as a learning project to understand NLP model training and inference.

---

## How it was built

- **Model:** T5-small  
- **Library:** Hugging Face Transformers  
- **Training:** Google Colab (GPU)  
- **Data:** Synthetic and private calendar events  

For privacy reasons, **no real calendar data or trained model weights are included** in this repository.

---

## Project structure

```text
calendar-ml-bot/
├── train.py          # Training script (from Colab)
├── inference.py      # Run the model on example events
├── tokenizer/        # Tokenizer configuration files
├── model/            # Model configuration files
├── data/             # Dataset documentation only
├── requirements.txt
├── .gitignore
└── README.md
