Intersectional Humor Perception

This code is for a Master's thesis studying whether intersectional age x gender differences shape humor word selection in human annotation, and whether LLMs (GPT-4o-mini, Claude Haiku 4.5, Gemini 2.5-flash) can replicate these patterns through sociodemographic prompting.

## Project structure

```
data/selected_216_words/       Human annotation data, 216 selected words
human_analysis/                Human-side analysis
llm_predictions/                Collected LLM outputs, one folder per model
  claude-haiku-4.5/
  gemini-2.5-flash/
  gpt-4o-mini/
    forced_choice/             Forced-choice selection rates per prompt style
    rating_scale/               1-5 rating scale outputs per prompt style
src/
  prediction/
    forced_choice/              Code that run the forced-choice LLM experiments with different prompt styles
      fc_direct_prompt.py
      fc_indirect_prompt.py
      fc_cot_prompt.py
    rating_scale/                Code that run the rating-scale LLM experiments with different prompt styles
      rs_direct_prompt.py
      rs_indirect_prompt.py
      rs_cot_prompt.py
  llm_analysis/                 Code for computing metrics on the LLM outputs
    forced_choice_dataset_1.py
    forced_choice_dataset_2.py
    rating_scale.py
  human_analysis/                Code for computing metrics on the human data
    intersectional_analysis.py       (6 intersectional age x gender groups)
    intersectional_analysis_gender.py (gender only, 2 groups)
    intersectional_analysis_age.py    (age only, 3 groups)
```

## Running the prediction scripts

Each script in `src/prediction/` sends prompts to one model for one dataset. Set your own API key in the `API_KEY` variable and model in the `MODEL`variable at the top of the script before running. Edit the `DATASET`/input-file variables near the top to switch between dataset 1 and dataset 2. This code ouputs the model wise csv files saved inside 

```bash
python3 src/prediction/forced_choice/fc_direct_prompt.py
```

## Running the analysis scripts

Each script in `src/llm_analysis/` and `src/human_analysis/` reads a CSV and prints all metrics straight to the console. Edit the `INPUT_FILE` and `HUMAN_FILE` variable near the top to point at the file you want to analyze.

```bash
python3 human_analysis/intersectional_analysis.py
```