import json

def fix_notebook(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    for cell in d['cells']:
        if 'source' not in cell: continue
        source = "".join(cell['source'])
        
        # 1. Scale boxes in evaluate_model
        if "def evaluate_model(" in source and "metric.update(preds, target)" in source:
            if "* IMG_SIZE" not in source:
                source = source.replace(
                    '\"boxes\": p_b[j].unsqueeze(0).to(device),',
                    '\"boxes\": p_b[j].unsqueeze(0).to(device) * 384.0,'
                )
                source = source.replace(
                    '\"boxes\": g_b[j].unsqueeze(0).to(device),',
                    '\"boxes\": g_b[j].unsqueeze(0).to(device) * 384.0,'
                )
                cell['source'] = [line + '\n' for line in source.split('\n')]
                if cell['source']: cell['source'][-1] = cell['source'][-1][:-1]

        # 2. Lower Learning Rate for phase 1 from 1e-2 to 1e-3
        # In fit_model_freeze_thaw
        if "def fit_model_freeze_thaw(" in source:
            source = source.replace("1e-2", "1e-3")
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source']: cell['source'][-1] = cell['source'][-1][:-1]
            
        # Also in hyperparameters constants
        if "HEAD_LR =" in source and "1e-2" in source:
            source = source.replace("1e-2", "1e-3")
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source']: cell['source'][-1] = cell['source'][-1][:-1]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)

fix_notebook("main_resnet18.ipynb")
fix_notebook("main_mobilenet_ssd.ipynb")
print("Fixes applied.")
