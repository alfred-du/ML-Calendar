# Model Configuration

This directory contains configuration files for the calendar event summarization model.

The model is based on a fine-tuned **T5-small** architecture and is trained to generate short, normalized summaries from free-form calendar event text.

---

## Files

- `config.json`  
  Defines the model architecture and core hyperparameters inherited from T5-small.

- `generation_config.json`  
  Specifies text generation parameters used during inference (e.g. max length, decoding behavior).

---

## Training

- Training was performed using the Hugging Face `Trainer` API.
- The base model was initialized from `t5-small`.
- Input format follows the T5 summarization convention using the `summarize:` prefix.

Example input: "Team standup meeting at 9am"
Example output: "Meeting"

---

## Model Artifacts

Trained model weights are intentionally **excluded** from this repository due to:
- File size constraints
- Data privacy considerations

The configuration files provided here allow the training setup to be fully understood and reproduced when paired with appropriate data and model artifacts.

---

## Notes

This directory is intended to document model behavior and configuration rather than serve as a complete distribution of the trained model.
