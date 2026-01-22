# Solar Output Prediction Agent

AI-powered agent for predicting solar panel energy output based on weather conditions.

## Quick Start

### Prerequisites
```bash
# Install dependencies
pip install pandas numpy scikit-learn joblib ollama

# Start Ollama (in a separate terminal)
ollama serve

# Pull the LLM model
ollama pull llama3.1:8b
```

### Required Files
Place these files in the same directory:
- `solar_agent.py` (the main script)
- `elasticnet_pvout_model.joblib` (your trained model)
- `dataset_train_only.csv` (training data for city averages)

## Usage

### Option 1: Interactive Mode (Recommended for Demos)
```bash
python solar_agent.py
```

Then ask questions naturally:
```
You: What's the expected PVOUT in Perth?
Agent: 🌞 PVOUT: 4.67 kWh/kWp/day
       Based on historical weather in Perth

You: Compare Sydney vs Darwin for 200 kWp system
Agent: 📊 Sydney vs Darwin
       🌞 Sydney: 4.19 kWh/kWp/day
          Daily: 670.29 kWh (200 kWp)
       🌞 Darwin: 4.68 kWh/kWp/day
          Daily: 749.50 kWh (200 kWp)
       🏆 Darwin is better by 0.50 kWh/kWp/day
```

### Option 2: Command Line (Single Query)
```bash
python solar_agent.py "What's the expected output for a 100 kWp farm in Perth?"
```

### Option 3: Import as Module
```python
from solar_agent import run_agent, format_response

result = run_agent("Expected PVOUT in Melbourne?")
print(format_response(result))

# Or get raw JSON
print(result)  # {"pvout_kwh_per_kwp_day": 3.97, ...}
```
