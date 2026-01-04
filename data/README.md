# Dataset Documentation

This directory documents the training data used for the calendar event summarization model.

---

## Overview

The model was trained on structured calendar event text paired with short summary labels.
Due to the sensitive nature of calendar data, **no raw datasets are included in this repository**.

---

## Data Source

- Combination of synthetic and privately generated calendar events
- No real personal calendar data is committed
- CSV files were used during training and validation

---

## Data Format

Each dataset record contains the following fields:

- `event_text`  
  Free-form text describing a calendar event.

- `summary`  
  Short, normalized label or summary for the event.

### Example (synthetic)

```csv
event_text,summary
"Team standup meeting at 9am","Meeting"
"Lunch with Sarah","Lunch"
"Dentist appointment","Dentist"
