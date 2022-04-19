from utils.model import Perceptron
import pandas as pd
from utils.all_utils import prepare_data

AND = {
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 0, 0, 1]
}

df = pd.DataFrame(AND)

df



X, y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model_AND = Perceptron(eta = ETA, epochs= EPOCHS)

model_AND.fit(X, y)

_ = model_AND.total_loss()