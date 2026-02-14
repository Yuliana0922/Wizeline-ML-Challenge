### ML Modeling Challenge for Wizeline
### Yuliana Gonzalez Baena

## Overview
This is my solution to the Wizeline ML Coding challenge, Received at Feb 12. 10:57AM.  

**Problem:**

Multivariate regression problem with 20 features and 800 training samples.

**Solution:** 

**Model:** Polynomial Features (Degree 3) + Lasso Regression  
**Features used:** 5 of 20 (Feature selection by correlation and tree selection)  
**Regularization:** Lasso (Alpha = 0/03)  
**Cross-Validation:** 5-fold with Pipeline

## Key insights
1. **Feature selection.**  
Only 5 of 20 features have predictive power:

- feature_2 (correlation: 0.55)
- feature_13 (correlation: 0.40)
- feature_9 (correlation: 0.36)
- feature_11 (correlation: 0.32)
- feature_18 (correlation: 0.06, but important non-linear contribution)  
The remaining 15 features are *mostly noise* and excluding them improves generalization.

    1.1. **Feature Selection Process**  
    - Correlation analysis identified 4 strongly correlated features  
    - Gradient Boosting feature importance confirmed these + revealed feature_18
    - Lasso on polynomial features validated that only terms involving these 5 features have non-zero coefficients

2. **Non-linear relationships**  
Linear regression on selected features achieves R² ≈ 0.70. The jump to R² ≈ 0.90 comes from polynomial features (degree 3), which capture:

- Squared terms: f18², f13², f9²
- Cubic interactions: f13·f9², f13²·f9

3. **Regularization**  
Lasso regression (alpha = 0.03) automatically selects ~32 of 55 polynomial terms, preventing overfitting on the small dataset.

## Methodology
**Pipeline Architecture:**  
All preprocessing is wrapped in a scikit-learn Pipeline to prevent data leakage during cross-validation.

## Results  

Linear Regression (5 features): CV R2 0.693  
Poly3 + Ridge: 0.901: CV R2 0.901  
**Poly3 + Lasso (alpha=0.03): CV R2 0.905** -  Winner :)  
Gradient Boosting (all 20 features): 0.849

## Overfitting check    
Training R2: 0.914  
CV R2 (out-of-fold): 0.905  
Gap: 0.009  
No indication of overfitting

## How to execute within Docker
Requirements: Have Docker installed  
1. Clone GitHub repo
2. Build Docker image by running  
``docker build -t wizelinechallenge .``
3. Execute commmand   ``docker run -it -p 8000:8000 -p 8888:8888 -v ${PWD}:/app wizelinechallenge``  
This command will execute a fastAPI server that you can execute by visit `http://localhost:8000/docs`
This will show you SwaggerGUI, where you'll be able to directly  
You can also call by Postman or CURL   
4. To execute the Jupyter notebook server, use the URL that Docker will provide. Ej. `http://127.0.0.1:8888/tree?token=your_token_here`

Author: Yuliana Gonzalez - Built for Wizeline Challenge