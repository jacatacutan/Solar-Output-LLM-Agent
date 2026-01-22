#!/usr/bin/env python3
"""
Solar Output Prediction Agent
Run with: python solar_agent.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import ollama
import json
import re

# ========== CONFIGURATION ==========
MODEL_PATH = Path("elasticnet_pvout_model.joblib")
DATA_PATH = Path("dataset_train_only.csv")
OLLAMA_MODEL = "llama3.1:8b"

# ========== LOAD MODEL AND DATA ==========
print("Loading model and data...")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURES = artifact["features"]

df = pd.read_csv(DATA_PATH)
df["temp_range"] = df["MaxTemp_mean"] - df["MinTemp_mean"]
df["humidity_avg"] = (df["Humidity9am_mean"] + df["Humidity3pm_mean"]) / 2
df["cloud_avg"] = (df["Cloud9am_mean"] + df["Cloud3pm_mean"]) / 2
df["pressure_avg"] = (df["Pressure9am_mean"] + df["Pressure3pm_mean"]) / 2
df["wind_speed_avg"] = (df["WindSpeed9am_mean"] + df["WindSpeed3pm_mean"]) / 2

city_weather = df.set_index("Location")[FEATURES].sort_index()

# Get imputation values from the trained model's imputer
if hasattr(model, 'named_steps') and 'imputer' in model.named_steps:
    imputer = model.named_steps['imputer']
    GLOBAL_MEDIANS = {feat: val for feat, val in zip(FEATURES, imputer.statistics_)}
    print(f"✓ Using imputation values from trained model")
else:
    # Fallback: calculate from data (less ideal)
    GLOBAL_MEDIANS = df[FEATURES].median(numeric_only=True).to_dict()
    print(f"⚠ Using calculated medians (model imputer not accessible)")

AVAILABLE_CITIES = sorted(city_weather.index.tolist())

print(f"✓ Model loaded")
print(f"✓ Feature order: {FEATURES}")
print(f"✓ Loaded data for {len(AVAILABLE_CITIES)} cities\n")

# ========== UTILITY FUNCTIONS ==========
def coerce_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", "")
    try:
        return float(s)
    except:
        return None

def normalize_city_name(city: str) -> str:
    if not city:
        return None
    if city in AVAILABLE_CITIES:
        return city
    city_lower = city.lower()
    for available in AVAILABLE_CITIES:
        if available.lower() == city_lower:
            return available
    return None

def validate_weather(weather: dict) -> tuple[bool, list]:
    warnings = []
    checks = {
        "WindGustSpeed_mean": (0, 200, "km/h"),
        "Rainfall_mean": (0, 500, "mm"),
        "Evaporation_mean": (0, 50, "mm"),
        "Sunshine_mean": (0, 14, "hours"),
        "temp_range": (0, 50, "°C"),
        "humidity_avg": (0, 100, "%"),
        "cloud_avg": (0, 9, "oktas"),
        "pressure_avg": (900, 1100, "hPa"),
        "wind_speed_avg": (0, 150, "km/h")
    }
    for feature, (min_val, max_val, unit) in checks.items():
        val = weather.get(feature)
        if val is not None:
            if val < min_val or val > max_val:
                warnings.append(f"{feature}={val} outside range [{min_val}-{max_val}] {unit}")
    return len(warnings) == 0, warnings

# ========== CORE PREDICTION ==========
def predict_pvout(weather: dict) -> float:
    missing = set(FEATURES) - set(weather.keys())
    if missing:
        raise ValueError(f"Missing: {missing}")
    row = pd.DataFrame([[weather[f] for f in FEATURES]], columns=FEATURES)
    return float(model.predict(row)[0])

def get_typical_weather(city: str) -> dict:
    normalized_city = normalize_city_name(city)
    if normalized_city is None:
        raise ValueError(f"Unknown city: '{city}'. Try: {', '.join(AVAILABLE_CITIES[:5])}")
    
    weather = city_weather.loc[normalized_city, FEATURES].to_dict()
    
    # Fill any NaN with model's imputation values (for cities with missing data)
    for feature in FEATURES:
        if pd.isna(weather[feature]):
            weather[feature] = GLOBAL_MEDIANS[feature]
    
    return weather

def fill_missing_weather(partial_weather: dict, city: str = None) -> dict:
    base = GLOBAL_MEDIANS.copy()
    if city:
        normalized_city = normalize_city_name(city)
        if normalized_city:
            try:
                base.update(get_typical_weather(normalized_city))
            except:
                pass
    out = {}
    for k in FEATURES:
        v = coerce_float(partial_weather.get(k))
        out[k] = v if v is not None else base[k]
    return out

def answer_query(city=None, weather=None, system_kwp=None, performance_ratio=0.8):
    if weather is None:
        if city is None:
            raise ValueError("Provide weather or city")
        weather = get_typical_weather(city)
    
    # Validate weather inputs and collect warnings
    is_valid, warnings = validate_weather(weather)
    pvout = predict_pvout(weather)
    
    result = {
        "pvout_kwh_per_kwp_day": round(pvout, 3),
        "assumptions": {
            "weather_source": "user_provided" if city is None else f"historical_average_{city}",
            "performance_ratio": performance_ratio,
        }
    }
    
    # Always include warnings if present (don't hide them)
    if warnings:
        result["warnings"] = warnings
    
    if system_kwp is not None:
        result["energy_kwh_day"] = round(pvout * system_kwp * performance_ratio, 2)
        result["system_kwp"] = system_kwp
    return result

# ========== LLM PARSING ==========
SYSTEM_PROMPT = f"""
You are a solar-output assistant.
Required features: {json.dumps(FEATURES)}
Available cities: {', '.join(AVAILABLE_CITIES[:10])}... ({len(AVAILABLE_CITIES)} total)
"""

def llm_parse(user_text: str) -> dict:
    # Quick check for obviously off-topic queries
    solar_keywords = ['pvout', 'solar', 'energy', 'kwp', 'kw', 'mw', 'output', 'panel', 
                      'farm', 'installation', 'system', 'production', 'generate', 'yield',
                      'expected', 'predict', 'compare', 'better', 'power']
    
    # Check for city names
    has_city = any(city.lower() in user_text.lower() for city in AVAILABLE_CITIES)
    has_solar_keyword = any(keyword in user_text.lower() for keyword in solar_keywords)
    
    # Check for system size mentions
    has_system_size = bool(re.search(r'\d+\s*(kWp|kW|MW)', user_text, re.IGNORECASE))
    
    # Check for weather parameters (numbers followed by weather-related words)
    has_weather_params = bool(re.search(r'\d+.*?(sunshine|humidity|temp|rain|evap|cloud|wind|pressure)', user_text, re.IGNORECASE))
    
    # Reject if it's clearly off-topic (city mentioned but no solar context)
    if has_city and not has_solar_keyword and not has_system_size and not has_weather_params:
        return {
            "intent": "off_topic",
            "message": "I can only answer questions about solar energy output. Try asking about PVOUT or energy production!"
        }
    
    # Reject vague queries with no context (like just "pvout" or "solar")
    if len(user_text.split()) <= 2 and not has_city and not has_system_size and not has_weather_params:
        return {
            "intent": "off_topic",
            "message": "Please provide more details! Specify a city (e.g., 'PVOUT in Perth') or weather conditions."
        }
    
    # Reject if no solar context at all
    if not has_city and not has_solar_keyword and not has_system_size and not has_weather_params:
        return {
            "intent": "off_topic",
            "message": "I'm a solar output prediction assistant. Ask me about PVOUT, energy output, or compare cities for solar installations!"
        }
    
    fallback_kwp = None
    kwp_match = re.search(r'(\d+(?:\.\d+)?)\s*kWp', user_text, re.IGNORECASE)
    kw_match = re.search(r'(\d+(?:\.\d+)?)\s*kW(?!p)', user_text, re.IGNORECASE)
    mw_match = re.search(r'(\d+(?:\.\d+)?)\s*MW', user_text, re.IGNORECASE)
    
    if kwp_match:
        fallback_kwp = float(kwp_match.group(1))
    elif mw_match:
        fallback_kwp = float(mw_match.group(1)) * 1000
    elif kw_match:
        fallback_kwp = float(kw_match.group(1))
    
    try:
        prompt = f"""
User query: {user_text}

Return ONLY valid JSON:
{{
  "intent": "pvout_by_city OR energy_by_city OR pvout_by_weather OR energy_by_weather OR compare_cities",
  "city": "string or null",
  "city2": "string or null",
  "system_kwp": "number or null",
  "weather": {{}}
}}

IMPORTANT: 
- Extract system_kwp from mentions of kWp/kW/MW
- Convert MW to kWp (multiply by 1000)
- If system size mentioned, intent should be energy_by_city or energy_by_weather

Examples:
- "2 MW installation in Adelaide" → {{"intent": "energy_by_city", "city": "Adelaide", "system_kwp": 2000}}
- "100 kWp in Perth" → {{"intent": "energy_by_city", "city": "Perth", "system_kwp": 100}}
"""
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": prompt.strip()},
            ],
            options={"temperature": 0.0},
        )
        content = resp["message"]["content"].strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        parsed = json.loads(content)
        
        intent = parsed.get("intent", "")
        if "|" in intent:
            intent = intent.split("|")[0].strip()
            parsed["intent"] = intent
        
        valid_intents = ["pvout_by_city", "energy_by_city", "pvout_by_weather", "energy_by_weather", "compare_cities"]
        if intent not in valid_intents:
            parsed["intent"] = "pvout_by_city"
        
        # CRITICAL FIX: Apply regex fallback and force intent update
        if fallback_kwp is not None:
            # Override LLM's extraction if regex found a value
            parsed["system_kwp"] = fallback_kwp
            
            # Force intent to energy_* since we have a system size
            if parsed["intent"] == "pvout_by_city":
                parsed["intent"] = "energy_by_city"
            elif parsed["intent"] == "pvout_by_weather":
                parsed["intent"] = "energy_by_weather"
        
        return parsed
    except Exception as e:
        raise RuntimeError(f"LLM parsing failed: {e}\nIs Ollama running?")

# ========== AGENT ==========
def run_agent(user_text: str) -> dict:
    try:
        plan = llm_parse(user_text)
    except Exception as e:
        return {"error": "llm_parsing_failed", "details": str(e)}
    
    intent = plan.get("intent")
    
    # Handle off-topic queries
    if intent == "off_topic":
        return {"error": "off_topic", "message": plan.get("message", "Please ask about solar output predictions.")}
    
    city = plan.get("city")
    city2 = plan.get("city2")
    system_kwp = coerce_float(plan.get("system_kwp"))
    pr = coerce_float(plan.get("performance_ratio")) or 0.8
    weather = plan.get("weather") or {}
    
    def validate_city(c, param_name="city"):
        if not c:
            raise ValueError(f"{param_name} required for '{intent}'")
        normalized = normalize_city_name(c)
        if not normalized:
            raise ValueError(f"Unknown {param_name}: '{c}'")
        return normalized
    
    try:
        if intent == "pvout_by_city":
            city = validate_city(city)
            return answer_query(city=city, performance_ratio=pr)
        
        elif intent == "energy_by_city":
            city = validate_city(city)
            if system_kwp is None:
                raise ValueError("system_kwp required")
            return answer_query(city=city, system_kwp=system_kwp, performance_ratio=pr)
        
        elif intent in ("pvout_by_weather", "energy_by_weather"):
            if intent == "energy_by_weather" and system_kwp is None:
                raise ValueError("system_kwp required")
            
            # Fill missing weather with city averages or global medians
            final_weather = fill_missing_weather(weather, city=city)
            
            return answer_query(weather=final_weather, system_kwp=system_kwp, performance_ratio=pr)
        
        elif intent == "compare_cities":
            city = validate_city(city, "city")
            city2 = validate_city(city2, "city2")
            r1 = answer_query(city=city, system_kwp=system_kwp, performance_ratio=pr)
            r2 = answer_query(city=city2, system_kwp=system_kwp, performance_ratio=pr)
            
            result = {
                "city1": city, "city2": city2,
                "result1": r1, "result2": r2,
                "delta_pvout": round(r1["pvout_kwh_per_kwp_day"] - r2["pvout_kwh_per_kwp_day"], 3)
            }
            if system_kwp and "energy_kwh_day" in r1:
                result["delta_energy_kwh_day"] = round(r1["energy_kwh_day"] - r2["energy_kwh_day"], 2)
            return result
        
        else:
            return {"error": "unknown_intent", "intent": intent}
    
    except Exception as e:
        return {"error": "execution_failed", "details": str(e)}

# ========== HUMAN-READABLE OUTPUT ==========
def format_response(result: dict) -> str:
    if "error" in result:
        if result.get("error") == "off_topic":
            return f"🤔 {result.get('message', 'Please ask about solar predictions!')}"
        return f"❌ Error: {result.get('details', result.get('error'))}"
    
    lines = []
    
    if "city1" in result:
        lines.append(f"📊 {result['city1']} vs {result['city2']}")
        r1, r2 = result['result1'], result['result2']
        lines.append(f"\n🌞 {result['city1']}: {r1['pvout_kwh_per_kwp_day']:.2f} kWh/kWp/day")
        if 'energy_kwh_day' in r1:
            lines.append(f"   Daily: {r1['energy_kwh_day']:.2f} kWh ({r1['system_kwp']:.0f} kWp)")
        lines.append(f"\n🌞 {result['city2']}: {r2['pvout_kwh_per_kwp_day']:.2f} kWh/kWp/day")
        if 'energy_kwh_day' in r2:
            lines.append(f"   Daily: {r2['energy_kwh_day']:.2f} kWh ({r2['system_kwp']:.0f} kWp)")
        delta = result['delta_pvout']
        winner = result['city1'] if delta > 0 else result['city2']
        lines.append(f"\n🏆 {winner} is better by {abs(delta):.2f} kWh/kWp/day")
    else:
        pvout = result.get('pvout_kwh_per_kwp_day')
        if 'energy_kwh_day' in result:
            lines.append(f"⚡ Daily Energy: {result['energy_kwh_day']:.2f} kWh")
            lines.append(f"   System: {result['system_kwp']:.0f} kWp")
            lines.append(f"   PVOUT: {pvout:.2f} kWh/kWp/day")
        else:
            lines.append(f"🌞 PVOUT: {pvout:.2f} kWh/kWp/day")
        
        source = result.get('assumptions', {}).get('weather_source', '')
        if source.startswith('historical_average_'):
            city = source.replace('historical_average_', '')
            lines.append(f"   Based on historical weather in {city}")
        elif source == 'user_provided':
            lines.append(f"   Based on your weather conditions")
        
        # Show warnings prominently
        if 'warnings' in result:
            lines.append("")
            lines.append("⚠️  WARNINGS:")
            for warning in result['warnings']:
                lines.append(f"   • {warning}")
            lines.append("   ⚠️  Prediction may be unreliable with out-of-range values")
    
    return "\n".join(lines)

# ========== INTERACTIVE MODE ==========
def interactive_mode():
    print("="*60)
    print("SOLAR OUTPUT PREDICTION AGENT")
    print("="*60)
    print("\nCommands:")
    print("  - Ask questions naturally")
    print("  - 'cities' - list available cities")
    print("  - 'examples' - show example queries")
    print("  - 'quit' - exit")
    print("\n" + "="*60 + "\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if query.lower() == 'cities':
                print(f"\nAvailable cities ({len(AVAILABLE_CITIES)}):")
                for i in range(0, len(AVAILABLE_CITIES), 5):
                    print("  " + ", ".join(AVAILABLE_CITIES[i:i+5]))
                print()
                continue
            if query.lower() == 'examples':
                print("\nExample queries:")
                print("  • What's the expected PVOUT in Perth?")
                print("  • 100 kWp solar farm in Sydney, daily energy?")
                print("  • Compare Melbourne vs Brisbane for 50 kWp system")
                print()
                continue
            
            result = run_agent(query)
            print("\nAgent:", format_response(result))
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

# ========== MAIN ==========
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Non-interactive mode: process command line query
        query = " ".join(sys.argv[1:])
        result = run_agent(query)
        print(format_response(result))
    else:
        # Interactive mode
        interactive_mode()