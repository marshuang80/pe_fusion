PARAMETERS = {
    'elasticnet': {
        "C": [1e-3, 1e-1, 1, 10, 1000], 
        "class_weight": ['balanced', None],
        "max_iter": [100, 500, 1000],
        "l1_ratio": [0.01, 0.1, 0.3, 0.5, 0.9, 0.99]
    }
}