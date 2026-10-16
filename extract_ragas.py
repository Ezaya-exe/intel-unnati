import pandas as pd
import json

df = pd.read_csv('data/evaluation/ragas_scores.csv')

# Count valid (non-NaN) rows
valid_relevancy = df.dropna(subset=['answer_relevancy'])

print(f"Total rows in CSV: {len(df)}")
print(f"Rows with valid answer_relevancy: {len(valid_relevancy)}")

if len(valid_relevancy) > 0:
    vals = valid_relevancy['answer_relevancy']
    print(f"Mean:   {vals.mean():.3f}")
    print(f"Median: {vals.median():.3f}")
    print(f"Min:    {vals.min():.3f}")
    print(f"Max:    {vals.max():.3f}")

# Same for faithfulness (probably all NaN, but check)
valid_faith = df.dropna(subset=['faithfulness'])
print(f"\nRows with valid faithfulness: {len(valid_faith)}")
if len(valid_faith) > 0:
    vals = valid_faith['faithfulness']
    print(f"Mean faithfulness: {vals.mean():.3f}")

# Save summary JSON
summary = {}
for m in ['faithfulness', 'answer_relevancy']:
    if m in df.columns:
        mean_val = df[m].mean()
        summary[m] = None if mean_val != mean_val else round(float(mean_val), 4)

with open('data/evaluation/ragas_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print("\nSummary saved to data/evaluation/ragas_summary.json")
print(json.dumps(summary, indent=2))