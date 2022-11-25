Installation:

```pip install -r requirements.txt```  
```pip install .```

Usage:

```python ml/train_pipeline.py configs/train_config.yaml```   
```python ml/predict_pipeline.py configs/test_config.yaml```

Tests:

```pytest tests/```

Check that pipeline logs results to mlflow with docker:   
```docker-compose up --build```
