"""Entry point for review comment C05; shared engine handles execution."""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'C05_component_ablation.json')
    parser.add_argument('--output', type=Path,
                        default=root / 'results' / ('C05_component_ablation_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    subprocess.run([sys.executable, str(root / 'shared_experiment_core.py'),
                    '--config', str(args.config), '--output', str(args.output)], check=True)
    subprocess.run([sys.executable, str(root / 'C04_C07_C08_result_summary.py'),
                    str(args.output)], check=True)


if __name__ == '__main__':
    main()
