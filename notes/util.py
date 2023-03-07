import argparse
from time import time
  
  
def timed_process(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


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
