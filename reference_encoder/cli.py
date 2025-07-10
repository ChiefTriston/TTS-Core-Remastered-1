import argparse

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--device', choices=['cpu','cuda'], default='cuda',
        help='Compute device for training/inference'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output-dir', default='./checkpoints',
        help='Directory to save checkpoints and logs'
    )
    return parser