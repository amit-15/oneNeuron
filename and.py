from utils.model import Perceptron
import pandas as pd
from utils.all_utils import prepare_data, save_model, save_plot
import logging


logging_str = "[%(asctime)s: %(levelname)s: %(module)s:] %(massage)s"
logging.basicConfig(level=logging.INFO, format=logging_str)

def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(data)
    logging.info(f"This is actual dataFrame{df}")

    X, y = prepare_data(df)

    model= Perceptron(eta = eta, epochs= epochs)
    model.fit(X, y)

    _ = model.total_loss() #dummy variable

    save_model(model, filename= modelName)
    save_plot(df, plotName, model)
    
    
if __name__ == "__main__":  # entry point
        
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 0, 0, 1]
        }
    
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    
    main(data= AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)