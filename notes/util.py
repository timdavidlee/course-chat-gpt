import argparse


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="whether to enable the scripts to run on gpu or not"
    )
    return parser

def print_banner(some_text: str):
    print("=" * 80)
    print(some_text)
    print("=" * 80)