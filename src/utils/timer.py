import time
from contextlib import contextmanager

class Timer:
    """Simple timing utility to profile pipeline steps."""
    
    def __init__(self):
        self.timings = {}
    
    @contextmanager
    def step(self, name: str):
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.timings[name] = elapsed_ms
    
    def report(self) -> dict:
        return self.timings
    
    def print_report(self):
        total = sum(self.timings.values())
        print("\n" + "=" * 50)
        print("LATENCY PROFILE")
        print("=" * 50)
        for name, ms in sorted(self.timings.items(), key=lambda x: -x[1]):
            pct = (ms / total) * 100 if total > 0 else 0
            print(f"  {name:30s} {ms:8.1f} ms  ({pct:5.1f}%)")
        print("-" * 50)
        print(f"  {'TOTAL':30s} {total:8.1f} ms")
        print("=" * 50 + "\n")