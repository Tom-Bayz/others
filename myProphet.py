import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from fbprophet import Prophet

class myProphet():
    
    def __init__(self,y):
        
        self.df = pd.DataFrame()
        self.df["ds"] = y.index
        self.df["y"] = y.values
        
        
    def fit(self):
        
        self.model = Prophet()
        self.model.fit(self.df)
        
        
        return self.model
        
    def predict(self,start,end):
        
        gap = (end - start).days
        
        future = pd.DataFrame(data=pd.date_range(start,end,freq="0.5H"),columns=["ds"])
        forecast = self.model.predict(future)
        pred = pd.Series(forecast["yhat"].values,
                         index=pd.to_datetime(forecast["ds"]))
        
        
        return pred
    
    
    
if __name__ == "__main__":
    
    x = np.linspace(0,2,100)
    time = pd.date_range(start="2020-01-01",end="2020-01-31",freq="0.5H")
    x = np.linspace(0,100,len(time))
    data = pd.Series(np.sin(x),index=time)
    
    
    model = myProphet(y=data)
    
    model.fit()
    
    import datetime as dt
    pred = model.predict(start=dt.datetime(2020,1,31),end=dt.datetime(2020,2,8))
    
    print(data)
    
    pd.plotting.register_matplotlib_converters()
    data.plot()
    pred.plot(label="prediction",color="red")
    plt.grid(True)
    plt.show()