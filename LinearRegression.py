def LinearRegression(x, y):
    """
    This function calculates the linear regression of input data
    
    Inputs:
        - X: list of x values
        - y: list of y values
    Return:
        - m: slope of the data
        - n: cross zero
    """
    import numpy as np
    
    X_b = np.c_[np.ones((len(x),1)), x]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    X_new = np.array([[0],[2]])
    X_new_b = np.c_[np.ones((2,1)), X_new]
    y_predict = X_new_b.dot(theta_best)

    x1, y1 = X_new[0], y_predict[0]
    x2, y2 = X_new[1], y_predict[1]

    m = (y2 - y1)/(x2 - x1)
    n = ((x2 - x1)*y1 - (y2 - y1)*x1)/(x2 - x1)
    
    return m, n

