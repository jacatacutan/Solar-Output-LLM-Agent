# Solar Output Prediction Agent

This project explores the use of machine learning and large language models (LLMs) to predict and explain solar power output based on expected weather conditions in Australian cities. It combines classical regression modeling with an LLM-powered agent that can answer natural-language questions about photovoltaic (PV) energy production.

The goal of this project is to demonstrate how structured weather data, statistical modeling, and modern LLM tooling can be integrated into a practical decision-support system for solar energy planning.

---

## Project Overview

The project is composed of three main components:

1. ** AU Solar data preparation** (from raster-based solar datasets)
2. **Predictive modeling** using weather regressors
3. **An LLM-powered agent** that answers solar output questions in natural language

The system predicts **PVOUT** (photovoltaic output, measured in kWh/kWp/day) and can translate this into expected daily energy production for a given solar installation size.

---

## 1. Solar Data Preparation (Raster Processing)

Solar power data was sourced from the **Global Solar Atlas**, provided as raster (GeoTIFF) files representing solar irradiance and output metrics across Australia.

Key steps:
- Raster files were processed using geospatial tools (Rasterio / GDAL)
- Solar values were extracted for the geographic area corresponding to each city
- For each city, solar output was estimated using the **average raster values within the city’s spatial footprint**

This produced a city-level estimate of solar output per unit capacity, aligned with the weather dataset.

> Assumption: The extracted solar raster data corresponds to the same year as the selected weather data.

---

## 2. Predictive Modeling with Weather Regressors

### Dataset
The model uses the **Australia Daily Weather Dataset (Kaggle)** as its primary source of regressors.  
Rather than predicting at a daily granularity, the weather data was aggregated by city to produce stable, interpretable features.

### Feature Engineering
Highly correlated variables were consolidated into derived features, including:
- Temperature range
- Average humidity
- Average cloud cover
- Average pressure
- Average wind speed

This reduced multicollinearity while preserving physical interpretability.

### Model Choice
Several models were evaluated using repeated cross-validation, including:
- Ridge Regression
- Lasso
- ElasticNet
- Tree-based models (Random Forest, Gradient Boosting, XGBoost)

**ElasticNet** was selected as the final model due to:
- Strong predictive performance
- Stability on small datasets
- Built-in regularization
- Direct coefficient interpretability

### Performance
The final ElasticNet model achieved:

- **CV MAE:** ~0.13 kWh/kWp/day  
- **CV R²:** ~0.76  

The model was trained on aggregated weather data, consistent with the assumption that historical yearly solar data is not available.

---

## 3. LLM-Powered Solar Output Agent

An LLM-powered agent was built on top of the trained regression model to enable natural-language interaction.

### What the Agent Does
The agent:
- Parses user questions using an LLM (via Ollama)
- Extracts intent, location, system size, and weather inputs
- Calls the trained regression model for predictions
- Returns both **numerical results** and **human-readable explanations**

### Supported Question Types
The agent can reliably answer the following categories of questions:

1. **PVOUT by city**  
   *“What is the expected PVOUT in Perth?”*

2. **Daily energy by city and system size**  
   *“If I install a 100 kWp solar farm in Perth, what is the expected daily output?”*

3. **PVOUT from explicit weather conditions**  
   *“Given the following weather conditions, what is my expected PVOUT?”*

4. **Energy output from explicit weather + system size**  
   *“Given this weather and a 50 kWp system, what is my expected daily energy?”*

5. **City comparisons**  
   *“Compare expected solar output between Sydney and Darwin for a 200 kWp system.”*

### Handling Missing Information
When user queries are incomplete, the agent follows explicit rules:

- If **weather inputs are missing**, the agent:
  - Uses historical city-level weather averages when a city is provided
  - Falls back to global medians if no city is specified
- If **system size is missing**, the agent returns PVOUT only
- If required inputs are completely absent, the agent asks for clarification

These assumptions are always surfaced in the agent’s response.

---

## Interpretability and Explainability

Because the final model is linear (ElasticNet), interpretability is handled via model coefficients rather than post-hoc methods like SHAP.

This enables:
- Direct reasoning about how changes in sunshine, humidity, or rainfall affect output
- Scenario-based explanations (e.g., “If sunshine increases by 10% but humidity rises by 5%…”)
- Transparent, auditable predictions suitable for planning and decision-making

---

## Example Usage

The project includes:
- An interactive CLI agent
- Human-readable responses (not just JSON)
- Support for exploratory “what-if” scenarios

Example:

If I install a 100 kWp system in Perth, what is my expected daily energy

Response:
⚡ Daily Energy: 373.81 kWh
System: 100 kWp
PVOUT: 4.67 kWh/kWp/day
Based on historical weather in Perth

## Key Takeaways

- Classical regression models remain powerful when paired with careful feature engineering
- LLMs are most effective as **routing and reasoning layers**, not as predictors
- Explicit assumptions and interpretability are critical for real-world energy applications
- Combining ML + LLMs enables systems that are both accurate and user-friendly

---

## Future Work

- Temporal modeling using daily weather forecasts
- Integration with real-time weather APIs
- Expansion to other geographies
- UI-based frontend (e.g., Streamlit, Gradio)

---

## Disclaimer

This project is for educational and exploratory purposes. All predictions are estimates and should not be used as the sole basis for financial or infrastructure decisions.

