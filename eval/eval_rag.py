import pandas as pd
import numpy as np
from tqdm import tqdm

from external.connect import get_qdrant_client



if __name__ == "__main__":
    client = get_qdrant_client()