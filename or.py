from utils.model import Perceptron
import pandas as pd
from utils.all_utils import prepare_data, save_model, save_plot


def main(data, eta, epochs, filename, plotFileName):
    
    df = pd.DataFrame(data)
    print(df)

    X, y = prepare_data(df)

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    model_OR= Perceptron(eta = eta, epochs= epochs)

    model_OR.fit(X, y)

    _ = model_OR.total_loss()

    save_model(model_OR, filename= filename)
    save_plot(df, plotFileName, model_OR)
    
if __name__ == "__main__":
    
    OR = {
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 1, 1, 1]
    }
    
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
        
    main(data= OR, eta=ETA, epochs=EPOCHS, filename= "or.model", plotFileName= "or.png")