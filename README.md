

Portfolio Optimization Models
=============================

This repository contains some of the code which I am using for a project on portfolio optimization models. The models I intend to investigate includes but not limited to (1) Mean Variance Optimization \[1\], (2) Enhanced Portfolio Optimization \[2\], (3) undecided.  

To Do
-----

This list is not set in stone but the items in _cursive_ is especially undecided.  
Whenever an item is completed is will be marked with a ~~strike~~.

*   **Abstract**

*   **Literature Review**
    *   Markowitz ~ Mean Variance Framework
        *   How does it work.
        *   Model implications.
        *   Critique of the model
    *   Enhanced Portfolio Optimization
        *   How does it work.
        *   Principal component analysis.
        *   Noise in correlation(/covariance) matrix.
        *   Model implications.
    *   _Undecided model_

*   **Data**

    *   Data Format and Structure
        *   Describe what is in the data.
        *   Asset types and classes.
        *   _Transform returns._
    *   Basic descriptive statistics

    *   Returns descriptives.
    *   Equal weighted portfolio.
    *   _Martingale portfolio._

*   **Methodology**
*   **Empirical Results**
*   **Discussion**
*   **Conclusion**

Repo Directory
--------------

This is an overview of the repo structure ~ will be updated accordingly.  
```
project
│   README.md
│
|─── codelib
│   │─   helpers.py 
│   │─   enhanced_portfolio_optimization.py 
│   └─   mean_variance_markowitz.py
│
|─── mean variance optimization
│   └─   mvo_methodology.ipynb
│
|─── enhanced portfolio optimization
│   │─   epo_methodology.ipynb 
│   └─   pca_methodology.ipynb
│
|─── empirical testing
│   └─   portfolio_comparison.ipynb

```

Bibliography
============

1.  Markowitz, Harry, Portfolio Selection, The Journal of Finance 1952, Vol. 7, No. 1, pp. 77-91
    
2.  Pedersen, Lasse Heje; Babu, Abhilash; Levine, Ari Financial Analysts Journal 2020Vol. 77, No. 2, pp. 124-151