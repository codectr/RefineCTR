![RefineCTR](https://github.com/codectr/RefineCTR/blob/main/RefineCTR.png)

The code and experimental results of《A Comprehensive Analysis and Evaluation of Feature Refinement Modules for CTR Prediction》.  

Click-through rate (CTR) prediction is widely used in academia and industry. Most CTR prediction models follow the same design paradigm: Feature Embedding (FE) layer, Feature Interaction (FI) layer, and Prediction layer. RefineCTR inserts existing feature refinement modules between FE and FI layer to boost basic CTR models performance, which also sumarizes and compares those modules' effectiveness comprehensively. 

As shown in the following figures: (A) and (b) inserting a single FR module between FR and FI layer for stacked and parallel CTR models; (C) assigning two separate FR modules for different FI sub-network to generate discriminate feature distributions for parallel models.



![The primary backbone structures of common CTR prediction models ](D:\code\RefineCTR\figure\refineCTR.png)



# Feature Refinement Modules

We extrct 14 FR modules from existing works. And FAL equires that the input data not anonymous. Therefore, we do not evaluate FAL in the experiments as Criteo dataset is anonymous.

| Year | Module  | Literature  |
| :--: | :-----: | :---------: |
| 2019 | FEN     | IFM         |
| 2019 | SENET   | FiBiNet     |
| 2020 | FWN     | NON         |
| 2020 | DFEN    | DIFM        |
| 2020 | DRM     | FED         |
| 2020 | FAL     | FaFM        |
| 2020 | VGate   | GateNet     |
| 2020 | BGate   | GateNet     |
| 2020 | SelfAtt | InterHAt    |
| 2021 | TCE     | ContextNet  |
| 2021 | PFFN    | ContextNet  |
| 2022 | GFRL    | MCRF        |
| 2022 | FRNet-V | FRNet       |
| 2022 | FRNet-B | FRNet       |

# Basic CTR Models

|      | Model         | Publication | Patterns     |
| ---- | ------------- | ----------- | ------------ |
| 1    | FM            | ICDM'10     | A SIngle     |
| 2    | DeepFM        | IJCAI'17    | B   SIngle   |
| 3    | DeepFM        | IJCAI'17    | C   Separate |
| 4    | CN (DCN)      | ADKDD'17    | A   SIngle   |
| 5    | DCN           | ADKDD'17    | B   SIngle   |
| 6    | DCN           | ADKDD'17    | C   Separate |
| 7    | AFN (AFN+)    | AAAI'20     | A   SIngle   |
| 8    | AFN+          | AAAI'20     | B   SIngle   |
| 9    | AFN+          | AAAI'20     | C   Separate |
| 10   | CN2 (DCNV2)   | WWW'21      | A   SIngle   |
| 11   | DCNV2         | WWW'21      | B   SIngle   |
| 12   | DCNV2         | WWW'21      | C   Separate |
| 13   | CIN (xDeepFM) | KDD'18      | A   SIngle   |
| 14   | xDeepFM       | KDD'18      | B   SIngle   |
| 15   | xDeepFM       | KDD'18      | C   Separate |
|      |               |             |              |
| 16   | NFM           | SIGIR'17    | A   SIngle   |
| 17   | FwFM          | WWW'18      | A   SIngle   |
| 18   | FiBiNET       | RecSys'19   | C   Separate |
| 19   | PNN           | ICDM'16     | A   SIngle   |
| 20   | HOFM          | NIPS'16     | A   SIngle   |
| 21   | AFM           | IJCAI'17    | A   SIngle   |
| 22   | FINT          | arXiv' 21   | A   SIngle   |

Including basic model, we can generate 308(22*14) augmented  models. Meanwhile, for parallel CTR model,  we can assigning different FR module for different FI sub-networks. Now we assign two same FR modules for different sub-networks.  

# Experiment Results





 More detailed results and experimental analysis will be continuously updated

# Get started



# Core Framework of RefineCTR
