from numpy import genfromtxt, array
from sklearn.metrics import f1_score


def get_f1_score(predicted_path,expected_path):

    y = array([ 1 if float(i) > 0 else 0 for i in genfromtxt(expected_path, delimiter=',') ])
    y_hat = genfromtxt(predicted_path, delimiter=',')

    assert len(y)==len(y_hat)
    
    return f1_score(y,y_hat), y, y_hat