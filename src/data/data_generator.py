from typing import Tuple
import random
from typing import List

def generate_data(value_range: Tuple[int, int], size: int) -> List[Tuple[str, str]]:
    data: List[str] = []
    labels: List[str] = []
    for _ in range(size):
        paren: bool = random.choice([True, False])
        term1: int = random.randint(value_range[0], value_range[1])
        term2: int = random.randint(value_range[0], value_range[1])
        if paren:
            data.append(f"({term1} + {term2})")
            labels.append(f"{term1 + term2}")
        else:
            data.append(f"{term1} + {term2}")
            labels.append(f"{term1 + term2}")
    return data, labels