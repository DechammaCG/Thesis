import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from datetime import datetime

API_KEY    = ""
client     = OpenAI(api_key=API_KEY)
MODEL      = "gpt-4o-mini"
SLEEP_TIME = 2
MAX_RETRY  = 5
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
    {'label': '18-29_Female', 'years': '1997 and 2008', 'pronoun': 'She'},
    {'label': '18-29_Male',   'years': '1997 and 2008', 'pronoun': 'He'},
    {'label': '30-49_Female', 'years': '1977 and 1996', 'pronoun': 'She'},
    {'label': '30-49_Male',   'years': '1977 and 1996', 'pronoun': 'He'},
    {'label': '50-70_Female', 'years': '1956 and 1976', 'pronoun': 'She'},
    {'label': '50-70_Male',   'years': '1956 and 1976', 'pronoun': 'He'},
]

results = {p['label']: {} for p in personas}
total_calls = len(personas) * len(words_216)
call_count  = 0

for persona in personas:
    for word in words_216:
        prompt = f"Imagine a person born between {persona['years']} in the United States, participating in a humor study. {persona['pronoun']} is shown a word and asked to rate how funny {persona['pronoun'].lower()} finds it on a scale of 1 to 5.\nWord: '{word}'\n1 = not funny at all\n2 = slightly funny\n3 = moderately funny\n4 = very funny\n5 = extremely funny\nThe participant must answer with only a single number (1, 2, 3, 4, or 5), and nothing else."
        success = False
        for attempt in range(MAX_RETRY):
            try:
                response = client.chat.completions.create(
                    model      = MODEL,
                    messages   = [{"role": "user", "content": prompt}],
                    max_tokens = 5
                )
                answer = response.choices[0].message.content.strip()
                if answer in ['1', '2', '3', '4', '5']:
                    results[persona['label']][word] = int(answer)
                    call_count += 1
                    success    = True
                    if call_count % 50 == 0:
                        pct = call_count / total_calls * 100
                        eta = (total_calls - call_count) * SLEEP_TIME / 60
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"{call_count}/{total_calls} ({pct:.1f}%) "
                              f"{persona['label']} | '{word}' - '{answer}'")
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
            print(f"Skipped {persona['label']}_{word} after {MAX_RETRY} attempts")
        time.sleep(SLEEP_TIME)

BASE_OUTPUT_DIR = find_project_root('llm_predictions')
OUTPUT_DIR = BASE_OUTPUT_DIR / "llm_predictions" / MODEL / "rating_scale"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rows = []
for word in words_216:
    row = {'word': word}
    for p in personas:
        row[p['label']] = results[p['label']].get(word, None)
    rows.append(row)

df_out = pd.DataFrame(rows)
output_path = OUTPUT_DIR / f"{MODEL}_rs_indirect_prompt_{DATASET}.csv"
df_out.to_csv(output_path, index=False)