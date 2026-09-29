import time
import random
import pandas as pd
from pathlib import Path
from openai import OpenAI
from datetime import datetime

API_KEY = ""
client     = OpenAI(api_key=API_KEY)
MODEL      = "gpt-4o-mini"
SLEEP_TIME = 2
MAX_RETRY  = 5
N_ROUNDS   = 10
DATASET    = "dataset1"

def find_project_root(folder):
    path = Path(__file__).resolve().parent
    while not (path / folder).exists():
        if path.parent == path:
            raise FileNotFoundError(f"Could not find a '{folder}' folder above {__file__}")
        path = path.parent
    return path

BASE_DATA_DIR = find_project_root("data")
DATA_DIR = BASE_DATA_DIR / "data" / "selected_216_words"
human_dataset = pd.read_csv(DATA_DIR / "human_annotation_dataset1.csv").set_index('word')
words_216 = human_dataset.index.tolist()

personas = [
    {'label': '18-29_Female', 'ages': list(range(18, 30)), 'sex': 'Female'},
    {'label': '18-29_Male',   'ages': list(range(18, 30)), 'sex': 'Male'},
    {'label': '30-49_Female', 'ages': list(range(30, 50)), 'sex': 'Female'},
    {'label': '30-49_Male',   'ages': list(range(30, 50)), 'sex': 'Male'},
    {'label': '50-70_Female', 'ages': list(range(50, 71)), 'sex': 'Female'},
    {'label': '50-70_Male',   'ages': list(range(50, 71)), 'sex': 'Male'},
]

results   = {p['label']: {w: [] for w in words_216} for p in personas}
ages_used = {p['label']: [] for p in personas}

total_calls = N_ROUNDS * 36 * len(personas)
call_count  = 0

for round_num in range(1, N_ROUNDS + 1):
    random.seed(round_num * 42)
    shuffled  = words_216.copy()
    random.shuffle(shuffled)
    sextuples = [shuffled[i:i+6] for i in range(0, 216, 6)]
    for persona in personas:
        random.seed(round_num + hash(persona['label']))
        age = random.choice(persona['ages'])
        for sext_idx, sextuple in enumerate(sextuples):
            prompt = f"You are a {age} year old {persona['sex']} from the United States. You are participating in a humor study. You will be shown 6 words and must pick the one you find funniest.\nWords: {', '.join(sextuple)}\nWhich single word do you find the funniest? Reply with only that one word, and nothing else."
            success = False
            for attempt in range(MAX_RETRY):
                try:
                    response = client.chat.completions.create(
                        model    = MODEL,
                        messages = [{"role": "user", "content": prompt}]
                    )
                    answer = response.choices[0].message.content.strip().lower()
                    words_lower = [w.lower() for w in sextuple]
                    if answer in words_lower:
                        original_word = sextuple[words_lower.index(answer)]
                        for w in sextuple:
                            selected = 1 if w == original_word else 0
                            results[persona['label']][w].append(selected)
                        ages_used[persona['label']].append(age)
                        call_count += 1
                        success    = True
                        if call_count % 50 == 0:
                            pct = call_count / total_calls * 100
                            eta = (total_calls - call_count) * SLEEP_TIME / 60
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                                  f"{call_count}/{total_calls} ({pct:.1f}%) "
                                  f"Round {round_num} | {persona['label']} age={age} | "
                                  f"sextuple {sext_idx+1} - '{answer}'")
                        break
                    else:
                        print(f"Invalid '{answer}' attempt {attempt+1}/{MAX_RETRY}")
                        time.sleep(5)

                except Exception as e:
                    print(f"Error attempt {attempt+1}: {e}")
                    if '429' in str(e):
                        print(f"Rate limit reached")
                        time.sleep(30)
                    elif '503' in str(e):
                        print(f"Server error")
                        time.sleep(60)
                    else:
                        time.sleep(10)
            if not success:
                print(f"Skipped round {round_num} {persona['label']} sextuple {sext_idx} after {MAX_RETRY} attempts")
            time.sleep(SLEEP_TIME)
llm_rates = {}
for p in personas:
    llm_rates[p['label']] = {}
    for word in words_216:
        votes = results[p['label']][word]
        llm_rates[p['label']][word] = round(sum(votes)/len(votes), 3) if votes else None

BASE_OUTPUT_DIR = find_project_root('llm_predictions')
OUTPUT_DIR = BASE_OUTPUT_DIR / "llm_predictions" / MODEL / "forced_choice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
llm_df = pd.DataFrame(llm_rates)
llm_df.index.name = 'word'
llm_df = llm_df.reset_index()
output_path = OUTPUT_DIR / f"{MODEL}_fc_direct_prompt_{DATASET}.csv"
llm_df.to_csv(output_path, index=False)


