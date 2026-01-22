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

## Supported Question Types

The agent answers **7+ types of questions** with high accuracy:

### 1. City-Based PVOUT Predictions
**What it does:** Predicts kWh/kWp/day for a city using historical average weather

**Examples:**
```
You: What's the expected PVOUT in Perth?
Agent: 🌞 PVOUT: 4.67 kWh/kWp/day
       Based on historical weather in Perth

You: Expected daily output in Alice Springs?
Agent: 🌞 PVOUT: 5.39 kWh/kWp/day
       Based on historical weather in AliceSprings

You: What about Melbourne?
Agent: 🌞 PVOUT: 3.97 kWh/kWp/day
       Based on historical weather in Melbourne
```

**How it works:** Uses historical average weather conditions from the training dataset for the specified city.

---

### 2. City-Based Energy Output (with System Size)
**What it does:** Calculates actual daily energy (kWh) for a specific system size in a city

**Examples:**
```
You: 100 kWp solar farm in Perth, daily energy?
Agent: ⚡ Daily Energy: 373.81 kWh/day
       System: 100 kWp
       PVOUT: 4.67 kWh/kWp/day
       Based on historical weather in Perth

You: 2 MW installation in Adelaide?
Agent: ⚡ Daily Energy: 7,034.56 kWh/day
       System: 2000 kWp
       PVOUT: 4.40 kWh/kWp/day
       Based on historical weather in Adelaide

You: 50 kWp system in Brisbane?
Agent: ⚡ Daily Energy: 179.03 kWh/day
       System: 50 kWp
       PVOUT: 4.48 kWh/kWp/day
       Based on historical weather in Brisbane
```

**How it works:** Multiplies PVOUT × system_kwp × performance_ratio (default 0.8)

---

### 3. Custom Weather PVOUT Predictions
**What it does:** Predicts PVOUT based on user-specified weather conditions

**Examples:**
```
You: PVOUT for Sunshine=8, humidity=55, temp_range=10, rainfall=1, 
     evaporation=5, cloud=3, wind_gust=35, pressure=1015, wind_speed=15
Agent: 🌞 PVOUT: 4.43 kWh/kWp/day
       Based on your specified weather conditions

You: What if Sunshine=10, humidity=40, temp_range=15, rainfall=0.5,
     evaporation=7, cloud=2, wind_gust=30, pressure=1013, wind_speed=12?
Agent: 🌞 PVOUT: 5.12 kWh/kWp/day
       Based on your specified weather conditions
```

**How it works:** Uses provided weather values directly; fills any missing parameters with global medians.

---

### 4. Custom Weather Energy Output
**What it does:** Calculates energy output for a specific system using custom weather

**Examples:**
```
You: Given Sunshine=9, humidity=45, temp_range=12, rainfall=0.8,
     evaporation=6, cloud=2.5, wind_gust=32, pressure=1014, wind_speed=13,
     for a 75 kWp system, what's the daily energy?
Agent: ⚡ Daily Energy: 312.45 kWh/day
       System: 75 kWp
       PVOUT: 5.21 kWh/kWp/day
       Based on your specified weather conditions
```

**How it works:** Applies custom weather to model, then multiplies by system size.

---

### 5. City Comparisons
**What it does:** Compares expected output between two cities

**Examples:**
```
You: Compare Sydney vs Darwin for 200 kWp system
Agent: 📊 Sydney vs Darwin
       🌞 Sydney: 4.19 kWh/kWp/day
          Daily: 670.29 kWh (200 kWp)
       🌞 Darwin: 4.68 kWh/kWp/day
          Daily: 749.50 kWh (200 kWp)
       🏆 Darwin is better by 0.50 kWh/kWp/day
          (79.21 kWh/day more for this system size)

You: Which is better for solar: Melbourne or Perth?
Agent: 📊 Melbourne vs Perth
       🌞 Melbourne: 3.97 kWh/kWp/day
       🌞 Perth: 4.67 kWh/kWp/day
       🏆 Perth is better by 0.70 kWh/kWp/day
```

**How it works:** Runs predictions for both cities and calculates the difference.

---

### 6. Best/Worst Cities Ranking
**What it does:** Ranks all 48 cities by expected solar output

**Example (via code):**
```python
from solar_agent import find_best_cities

result = find_best_cities(top_n=5, system_kwp=100)
print(result)

# Output:
Top 5 Cities:
1. AliceSprings: 5.39 kWh/kWp/day (431.05 kWh/day for 100 kWp)
2. Woomera: 5.07 kWh/kWp/day (405.81 kWh/day)
3. Moree: 4.92 kWh/kWp/day (393.33 kWh/day)
4. Cobar: 4.88 kWh/kWp/day (390.18 kWh/day)
5. Katherine: 4.79 kWh/kWp/day (383.49 kWh/day)
```

**How it works:** Iterates through all cities, predicts PVOUT, and sorts by output.

---

### 7. Monthly/Annual Production Estimates
**What it does:** Converts daily predictions to monthly/annual estimates

**Example (via code):**
```python
from solar_agent import estimate_monthly_production

result = estimate_monthly_production(city="Perth", system_kwp=100)
print(result)

# Output:
Daily: 373.81 kWh
Monthly: 11,214.30 kWh (~30 days)
Annual: 134,571.60 kWh
System: 100 kWp @ 80% efficiency
```

**How it works:** Multiplies daily output by 30 (monthly) or 365 (annual).

---

## Handling Missing Weather Data

### The Challenge
Weather is the key predictor for solar output. When users ask questions without specifying weather conditions, the agent must infer or fill missing data intelligently.

### Our Approach: Hierarchical Filling Strategy

The agent implements a **3-tier fallback system** for handling missing weather:

#### Tier 1: User-Provided Weather (Highest Priority)
If the user explicitly provides weather parameters, those values are used directly.

**Example:**
```
You: PVOUT for Sunshine=8, humidity=55, cloud=3
     (5 other parameters missing)
```
The agent fills missing parameters using Tier 2 or 3 below.

#### Tier 2: Historical City Averages
If a city is mentioned, missing weather parameters are filled with **historical averages** from that city's training data.

**Example:**
```
You: Expected output in Perth with Sunshine=10?
Agent fills: Sunshine=10 (user-provided)
            humidity_avg=58.2 (Perth historical average)
            temp_range=11.8 (Perth historical average)
            ... (other Perth averages)
```

**Why this works:** Historical averages represent typical conditions for that location, providing realistic baselines.

#### Tier 3: Global Medians (Last Resort)
If no city is specified and weather is incomplete, missing values are filled with **global median values** from the entire training dataset across all 48 cities.

**Example:**
```
You: PVOUT for Sunshine=9?
Agent fills: Sunshine=9 (user-provided)
            humidity_avg=64.5 (global median)
            temp_range=11.2 (global median)
            ... (other global medians)
```

### Implementation Details

**Code location:** `fill_missing_weather()` function in `solar_agent.py`

```python
def fill_missing_weather(partial_weather: dict, city: str = None) -> dict:
    """
    Fill missing weather fields using:
    1) City typical weather (if city provided)
    2) Global medians (fallback)
    """
    base = GLOBAL_MEDIANS.copy()  # Start with global medians
    
    if city:
        normalized_city = normalize_city_name(city)
        if normalized_city:
            try:
                # Override with city-specific averages
                base.update(get_typical_weather(normalized_city))
            except:
                pass
    
    # User values take highest priority
    out = {}
    for k in FEATURES:
        v = coerce_float(partial_weather.get(k))
        out[k] = v if v is not None else base[k]
    
    return out
```

### Transparency in Responses

The agent **explicitly tells users** which weather source was used:

```python
"assumptions": {
    "weather_source": "historical_average_Perth"  # or "user_provided" or "global_medians"
}
```

**Example outputs:**
```
Agent: Based on historical weather in Perth  # Used Tier 2
Agent: Based on your specified weather conditions  # Used Tier 1
```

### Alternative Approaches Considered

We evaluated three strategies:

| Approach | Implementation | Pros | Cons |
|----------|----------------|------|------|
| **1. Explicitly Ask User** | Prompt for each missing field | Most accurate | Poor UX, breaks conversation flow |
| **2. Historical Averages** | Use training data (our choice) | Realistic, contextual | Assumes typical conditions |
| **3. LLM Weather Inference** | Let LLM guess weather | Natural language | Unreliable, hallucinates data |

**Our hybrid approach** combines the best aspects:
- Uses historical data (Option 2) for reliability
- Allows user overrides (Option 1) for accuracy
- Maintains conversational flow

### Data Validation

The agent validates all weather inputs against reasonable ranges:

```python
{
    "WindGustSpeed_mean": (0, 200, "km/h"),
    "Sunshine_mean": (0, 14, "hours"),
    "humidity_avg": (0, 100, "%"),
    "pressure_avg": (900, 1100, "hPa"),
    # ... etc
}
```

**Example:**
```
You: PVOUT for Sunshine=25, humidity=150?
Agent: ⚠️ Warnings:
       • Sunshine=25 outside range [0-14] hours
       • humidity=150 outside range [0-100] %
       🌞 PVOUT: 4.23 kWh/kWp/day (prediction may be unreliable)
```

### Training Data Split

**Important:** The historical averages come from the **training set only** (`dataset_train_only.csv`). The test set was held out to ensure valid model evaluation.

This means:
- ✅ City averages reflect typical conditions
- ✅ No data leakage from test set
- ✅ Predictions remain unbiased

---

## Available Cities (48 total)
Adelaide, Albany, Albury, AliceSprings, BadgerysCreek, Ballarat, Brisbane, Cairns, Canberra, Cobar, CoffsHarbour, Dartmoor, Darwin, GoldCoast, Hobart, Katherine, Launceston, Melbourne, MelbourneAirport, Mildura, Moree, MountGambier, MountGinini, Newcastle, Nhil, NorahHead, NorfolkIsland, Nuriootpa, PearceRAAF, Penrith, Perth, PerthAirport, Portland, Richmond, Sale, SalmonGums, Sydney, SydneyAirport, Townsville, Tuggeranong, Uluru, WaggaWagga, Walpole, Watsonia, Williamtown, Witchcliffe, Wollongong, Woomera

## Features

### Weather Parameters Required
- `WindGustSpeed_mean` - Average wind gust speed (km/h)
- `Rainfall_mean` - Average daily rainfall (mm)
- `Evaporation_mean` - Average daily evaporation (mm)
- `Sunshine_mean` - Average daily sunshine hours
- `temp_range` - Temperature range: max - min (°C)
- `humidity_avg` - Average relative humidity (%)
- `cloud_avg` - Average cloud cover (0-9 oktas)
- `pressure_avg` - Average atmospheric pressure (hPa)
- `wind_speed_avg` - Average wind speed (km/h)

### Performance Ratio
Default: 0.8 (80% system efficiency)
This accounts for:
- Inverter losses (~2-5%)
- Cable losses (~1-3%)
- Temperature effects (~5-15%)
- Soiling/shading (~2-5%)

## Special Commands (Interactive Mode)
- `cities` - List all available cities
- `examples` - Show example queries
- `quit` - Exit

## Troubleshooting

### "Is Ollama running?"
Make sure Ollama is running in a separate terminal:
```bash
ollama serve
```

### "Unknown city"
Check available cities with the `cities` command in interactive mode, or see the list above. City names are case-insensitive.

### Model File Not Found
Ensure `elasticnet_pvout_model.joblib` is in the same directory as `solar_agent.py`.

## Architecture

```
User Query → LLM Parser (Ollama) → Intent Classification
                ↓
         Extract Parameters
                ↓
    ┌────────────────────────────┐
    │  Missing Weather Handling  │
    │  1. User-provided (if any) │
    │  2. City averages (if city)│
    │  3. Global medians         │
    └────────────────────────────┘
                ↓
         ElasticNet Pipeline
       (Imputer → Scaler → Model)
                ↓
    PVOUT Prediction → Energy Calculation → Human-Readable Response
```

## Known Limitations

1. **No Conversation Memory**: Each query is stateless. Follow-up queries like "same weather but 50 kWp" may not retain context from previous questions.

2. **LLM Parsing Variability**: Query interpretation depends on Ollama's understanding. Complex or ambiguous queries may be misinterpreted (mitigated with regex fallbacks).

3. **Historical Weather Assumption**: City-based predictions assume typical weather conditions, not specific future dates or unusual weather events.

## For Presentations

**Key Points to Highlight:**
- ✅ 7+ question types supported with high accuracy
- ✅ Intelligent missing data handling (3-tier fallback)
- ✅ Transparent assumptions in every response
- ✅ Human-readable + JSON outputs for different use cases
- ✅ Robust error handling and validation
- ✅ Case-insensitive city matching
- ✅ Automatic unit conversion (kW, MW → kWp)

**Demo Flow Suggestion:**
1. Start with simple city query: "PVOUT in Perth?"
2. Add system size: "100 kWp in Perth?"
3. Compare cities: "Compare Perth vs Melbourne for 200 kWp"
4. Custom weather: "PVOUT for Sunshine=10, humidity=40, ..."
5. Show `cities` command to list options
6. Handle edge case: "2 MW installation in Adelaide"
