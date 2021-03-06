
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 


from sklearn.tree import DecisionTreeClassifier


model_fname = "model.save"
MODEL_NAME = "decision_tree_sklearn"

COST_THRESHOLD = float('inf')


# class InfCostStopCallback(Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         loss_val = logs.get('loss')
#         if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
#             print("Cost is inf, so stopping training!!")
#             self.model.stop_training = True


class DecisionTree_sklearn(): 
    
    def __init__(self, min_samples_split = 2, min_samples_leaf = 1, **kwargs) -> None:
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.model = self.build_model()     
        
        
    def build_model(self): 
        model = DecisionTreeClassifier(min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf)
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self.model, os.path.join(model_path, model_fname))
        


    @classmethod
    def load(cls, model_path):         
        dtclassifier = joblib.load(os.path.join(model_path, model_fname))
        # print("where the load function is getting the model from: "+ os.path.join(model_path, model_fname))        
        return dtclassifier


def save_model(model, model_path):
    # print(os.path.join(model_path, model_fname))
    joblib.dump(model, os.path.join(model_path, model_fname)) #this one works
    # print("where the save_model function is saving the model to: " + os.path.join(model_path, model_fname))
    

def load_model(model_path): 
    try: 
        model = joblib.load(os.path.join(model_path, model_fname))   
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


