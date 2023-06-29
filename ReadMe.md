## NMF_CR

NMF-CR is a model to implement link prediction based on non-negative matrix factorization incorporate row space information and column space information.

### Instruction of Python implementation

(1) NMF_CR_general.py is a general version of NMF_CR, its loss function is defined as follows:

![LCR](https://github.com/AnonSimRank/NMF-CR/blob/main/LCR.png)

when $\alpha_{0}=1$, $\alpha_{1}=1$, $\beta_{0}=1$ , NMF_CR_general.py degenerates into NMF_CR.py

(2) utils.py contains some common used functions, e.g., train_test_split to split dataset into training data and probe data.

(3) Metrics.py contains common used metrics.

### Example

Take the dataset `ASongIceFire` as an example, 

`CR_train_ASongIceFire_`*.`py`  implement NMF_CR(combined row space information and column space information) on dataset `ASongIceFire`  as low dimensionality $k$ varies.

`R_train_ASongIceFire_`*.`py`  implement NMF_R( row space information ) on dataset `ASongIceFire`  as low dimensionality $k$ varies.

`C_train_ASongIceFire_`*.`py`  implement NMF_C( column space information ) on dataset `ASongIceFire`  as low dimensionality $k$ varies.

`train_ASongIceFire_`*.`py`  implement original NMF on dataset `ASongIceFire`  as low dimensionality $k$ varies.

Folder `output` contain the results of all models.

For example, `python CR_train_ASongIceFire_k256.py`

