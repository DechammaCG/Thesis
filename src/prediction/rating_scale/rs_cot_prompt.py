import re
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
    {'label': '18-29_Female', 'age': 25, 'sex': 'Female'},
    {'label': '18-29_Male',   'age': 25, 'sex': 'Male'},
    {'label': '30-49_Female', 'age': 38, 'sex': 'Female'},
    {'label': '30-49_Male',   'age': 38, 'sex': 'Male'},
    {'label': '50-70_Female', 'age': 58, 'sex': 'Female'},
    {'label': '50-70_Male',   'age': 58, 'sex': 'Male'},
]

results = {p['label']: {} for p in personas}

def extract_rating(text):
    match = re.search(r'answer[:\s]+([1-5])', text.lower())
    if match:
        return match.group(1)
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines and lines[-1] in ['1', '2', '3', '4', '5']:
        return lines[-1]
    digits = re.findall(r'[1-5]', text)
    if digits:
        return digits[-1]
    return None

total_calls = len(personas) * len(words_216)
call_count  = 0

for persona in personas:
    for word in words_216:
        prompt = f"You are a {persona['age']} year old {persona['sex']} from the United States. You are participating in a humor study. You will be shown a word and must rate how funny you find it.\nWord: '{word}'\nThink briefly about why a person of your age and gender would find this word funny or not funny, considering your generational context and humor preferences. Then give your final rating.\n1 = not funny at all\n2 = slightly funny\n3 = moderately funny\n4 = very funny\n5 = extremely funny\nReasoning: [your brief reasoning]\nAnswer: [single number 1-5 only]"
        success = False
        for attempt in range(MAX_RETRY):
            try:
                response = client.chat.completions.create(
                    model      = MODEL,
                    messages   = [{"role": "user", "content": prompt}],
                    max_tokens = 300
                )
                raw    = response.choices[0].message.content.strip()
                answer = extract_rating(raw)
                if answer:
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
                    print(f"Could not extract from: '{raw[:80]}' attempt {attempt+1}/{MAX_RETRY}")
                    time.sleep(5)
            except Exception as e:
                print(f"Error attempt {attempt+1}: {e}")
                if '429' in str(e):
                    print(f"Rate limit reached")
                    time.sleep(30)
                elif '503' in str(e):
                    print(f"Server error.")
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
output_path = OUTPUT_DIR / f"{MODEL}_rs_cot_prompt_{DATASET}.csv"
df_out.to_csv(output_path, index=False)