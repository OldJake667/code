"""Run VMamba training locally and save an accuracy-only figure."""
import csv
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fine-tuned-only', action='store_true', help='Use a completed scratch run and train only its classifier.')
    args = parser.parse_args()
    if args.fine_tuned_only and not (ROOT / 'results_from_scratch/result_summary.json').is_file():
        parser.error('Finish the scratch run first; result_summary.json is missing.')
    notebook = json.loads((ROOT / 'VMamba_from_scratch.ipynb').read_text())
    namespace = {'__name__': '__main__'}
    for index, cell in enumerate(notebook['cells'], 1):
        if args.fine_tuned_only and index in (3, 4):
            continue
        if cell['cell_type'] == 'code':
            print(f'Running notebook cell {index}', flush=True)
            exec(compile(''.join(cell['source']), f'VMamba:cell-{index}', 'exec'), namespace)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, folder, title in zip(axes, ['results_from_scratch', 'results_fine_tuned'], ['VMamba-T from scratch', 'Fine-tuned VMamba — classifier only']):
        with (ROOT / folder / 'training_history.csv').open() as file:
            rows = list(csv.DictReader(file))
        epochs = [int(row['epoch']) for row in rows]
        for key, label in [('train_accuracy', 'Training'), ('validation_accuracy', 'Validation')]:
            values = [float(row[key]) for row in rows]
            best = max(range(len(values)), key=values.__getitem__)
            line, = ax.plot(epochs, values, label=f'{label}: max {values[best]:.1%} (epoch {epochs[best]})')
            ax.scatter(epochs[best], values[best], color=line.get_color())
        ax.set(title=title, xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1.05))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    output = ROOT / 'vmamba_accuracy.png'
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f'Completed. Accuracy graph: {output}', flush=True)


if __name__ == '__main__':
    main()
