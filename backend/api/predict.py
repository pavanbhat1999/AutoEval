from backend.api.preprocessing import preprocessing,preprocessing_bow_features,preprocessing_bow
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
class Model:
    def __init__(self):
        pass
    def training(self):
        X_train,X_test,y_train,y_test = preprocessing()

        print("\n Shapes of content ",X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    def predict(self,answer):
        
        return("#################   Marks given by Composite should Display#################################")





###############################   dummy model not to be used######################### 
class dummy_model:
    def __init__(self):
        pass
    def training(self):
        scores=[0,0,0]
        j=0
        for i in [ preprocessing, preprocessing_bow,preprocessing_bow_features]:
            print("Training Started........for...",i)
            start_time = time.time()
            X_train,X_test,y_train,y_test = i()
            print("\n Shapes of content ",X_train.shape,X_test.shape)
            
            alphas = np.array([3, 1, 0.3, 0.1, 0.3])

            lasso_regressor = Lasso()

            grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
            grid.fit(X_train, y_train)
            print("Training time taken for ",i,time.time()-start_time)
            y_pred = grid.predict(X_test)

            # summarize the results of the grid search
            print(grid.best_score_)
            print(grid.best_estimator_.alpha)

            # The mean squared error
            
            print(i,"Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
            scores[j] = grid.score(X_test, y_test)
            j=j+1
            # Explained variance score: 1 is perfect prediction
            print(i,'Variance score: %.2f' % grid.score(X_test, y_test))

            # Cohen’s kappa score: 1 is complete agreement
            print(i,'Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))
        print(scores)
    def predict(self,answer):
        
        return("sad")
##############################  dummy model end############################## 

#############################  driver code############################## 
def execute(answer):

    
    y_pred = answer
    
    # Real model which should return marks
    # model = Model()
    # prediction = model.predict(y_pred)
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    #  jsut a dummy model for testing and returning 
    model = dummy_model()
    model.training()
    prediction = model.predict(y_pred)  
    print("answershape:",answer.shape)
    
    return(prediction)

