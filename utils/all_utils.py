import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import logging

def prepare_data(df):
  """it is used to separate the dependent and independent features

  Args:
      df (pd.DataFrame): its the pandas DataFrame to

  Returns:
      tuple: its returns the tuple of the dependent and independent variable
  """
  logging.info("preparing the data by segregating the dependent and independent variable")
  
  X = df.drop("y", axis= 1)

  y= df["y"]

  return X, y


def save_model(model, filename):
  """This saves the trained model

  Args:
      model (python object): trained model to save
      filename (str): path to save the trained model
  """
  logging.info("saving the trained model")
  
  model_dir = "models"
  os.makedirs(model_dir, exist_ok= True) # ONLY CREATED IF MODEL_DIR DOSEN'T EXITS
  filePath = os.path.join(model_dir, filename) # model/filename
  joblib.dump(model, filePath)
  logging.info(f"saved the trained model{filePath}")  
  
def save_plot(df, file_name, model):
  """its saves the plot of trained model

  Args:
      df (dataFrame): _description_
      file_name (_type_): path to save the plot
      model : trained model
  """
   
  def _create_base_plot(df):
    logging.info("creating base plot of trained model")
    df.plot(kind = "scatter", x= "x1", y= "x2", c= "y", s= 100, cmap= "winter")
    plt.axhline(y= 0, color= "black", linestyle= "--", linewidth= 1)
    plt.axvline(x= 0, color= "black", linestyle= "--", linewidth= 1)
    figure= plt.gcf()
    figure.set_size_inches(10,8)

  def _plot_decision_regions(X, y, classfier, resolution = 0.02):
    logging.info("ploting the decision regions")
    colors = ("red", "blue", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    X= X.values #as a array
    x1 = X[:, 0]
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1, x1.max() +1
    x2_min, x2_max = x2.min() -1, x2.max() +1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z= classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z= Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.2, cmap= cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.xlim(xx2.min(), xx2.max())

    plt.plot()

  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok= True) # ONLY CREATED IF MODEL_DIR DOSEN'T EXITS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)
  logging.info(f"saving the plot at {plotPath}")
  