import json

for nb_name in ['new_exp.ipynb', 'new_exp_2.ipynb', 'new_exp_8b.ipynb']:
    nb = json.load(open(f'd:/code/sop/{nb_name}', encoding='utf-8'))
    print(f'=== {nb_name} ===')
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if any(k in src for k in ['_essays.jsonl', '_scores.json', 'layer_scores', 'ESSAY_FILE', 'savefig', 'shutil.copy']):
            for line in src.split('\n'):
                if any(k in line for k in ['ESSAY_FILE', '_essays', '_scores', 'layer_scores', 'savefig', 'shutil', 'json.dump']):
                    print(f'  Cell {i}: {line.strip()[:120]}')
    print()
