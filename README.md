![RefineCTR](https://github.com/codectr/RefineCTR/blob/main/RefineCTR.png)

The code and experimental results of《A Comprehensive Analysis and Evaluation of Feature Refinement Modules for CTR Prediction》.  

Click-through rate (CTR) prediction is widely used in academia and industry. Most CTR prediction models follow the same design paradigm: Feature Embedding (FE) layer, Feature Interaction (FI) layer, and Prediction layer. RefineCTR inserts existing feature refinement modules between FE and FI layer to boost basic CTR models performance, which also sumarizes and compares those modules' effectiveness comprehensively. 

As shown in the following figures: (A) and (B) inserting a single FR module between FR and FI layer for stacked and parallel CTR models; (C) assigning two separate FR modules for different FI sub-network to generate discriminate feature distributions for parallel models.



![The primary backbone structures of common CTR prediction models ](https://github.com/codectr/RefineCTR/blob/main/refineCTR%20framework.png)



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

We will continue to update and upload the latest experimental results and analysis.

## Criteo





## Frappe

| Modules   | SKIP   | FEN    | SENET  | FWN    | DFEN   | DRM    | VGate  | BGate  | SelfAtt | TCE    | PFFN   | GFRL   | FRNet-V | FRNet-B |
| --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- | ------ | ------ | ------ | ------- | ------- |
| FM        | 0.9786 | 0.9789 | 0.9800 | 0.9808 | 0.9799 | 0.9820 | 0.9801 | 0.9803 | 0.9806  | 0.9800 | 0.9822 | 0.9821 | 0.9828  | 0.9831  |
| DeepFM    | 0.9824 | 0.9828 | 0.9827 | 0.9824 | 0.9824 | 0.9827 | 0.9828 | 0.9825 | 0.9831  | 0.9824 | 0.9830 | 0.9828 | 0.9837  | 0.9840  |
| DeepFM(2) | 0.9824 | 0.9830 | 0.9829 | 0.9829 | 0.9827 | 0.9825 | 0.9835 | 0.9828 | 0.9836  | 0.9839 | 0.9829 | 0.9843 | 0.9848  | 0.9846  |
| CN        | 0.9797 | 0.9829 | 0.9798 | 0.9810 | 0.9810 | 0.9803 | 0.9803 | 0.9803 | 0.9816  | 0.9819 | 0.9826 | 0.9827 | 0.9825  | 0.9826  |
| DCN       | 0.9825 | 0.9830 | 0.9822 | 0.9826 | 0.9838 | 0.9834 | 0.9829 | 0.9820 | 0.9829  | 0.9827 | 0.9828 | 0.9838 | 0.9838  | 0.9837  |
| DCN(2)    | 0.9825 | 0.9834 | 0.9829 | 0.9831 | 0.9843 | 0.9843 | 0.9835 | 0.9829 | 0.9832  | 0.9839 | 0.9838 | 0.9840 | 0.9844  | 0.9847  |
| AFN       | 0.9812 | 0.9826 | 0.9812 | 0.9816 | 0.9822 | 0.9821 | 0.9821 | 0.9814 | 0.9820  | 0.9826 | 0.9815 | 0.9835 | 0.9838  | 0.9838  |
| AFN+      | 0.9827 | 0.9838 | 0.9827 | 0.9831 | 0.9840 | 0.9836 | 0.9830 | 0.9826 | 0.9830  | 0.9836 | 0.9827 | 0.9838 | 0.9843  | 0.9844  |
| AFN+(2)   | 0.9827 | 0.9840 | 0.9830 | 0.9840 | 0.9846 | 0.9838 | 0.9839 | 0.9827 | 0.9837  | 0.9838 | 0.9834 | 0.9841 | 0.9844  | 0.9847  |
| CN2       | 0.9810 | 0.9822 | 0.9813 | 0.9826 | 0.9830 | 0.9825 | 0.9827 | 0.9813 | 0.9827  | 0.9821 | 0.9817 | 0.9825 | 0.9826  | 0.9834  |
| DCNV2     | 0.9830 | 0.9833 | 0.9835 | 0.9831 | 0.9839 | 0.9837 | 0.9833 | 0.9826 | 0.9829  | 0.9833 | 0.9831 | 0.9840 | 0.9839  | 0.9845  |
| DCNV2(2)  | 0.9830 | 0.9838 | 0.9838 | 0.9838 | 0.9844 | 0.9838 | 0.9837 | 0.9828 | 0.9832  | 0.9841 | 0.9835 | 0.9845 | 0.9841  | 0.9849  |
| IMP       |        | 0.10%  | 0.04%  | 0.08%  | 0.12%  | 0.11%  | 0.08%  | 0.02%  | 0.09%   | 0.11%  | 0.10%  | 0.17%  | 0.20%   | 0.22%   |



# Get started

1. **Test existing model with existing  module**.
```
cd evaluation/mains
CUDA_VISIBLE_DEVICES=0 python main_frappe_base --model 0 --module 0
```

2. **Adding new model or module.**

   Our framework RefineCTR is modularized, users can adjust or add basic models and modules easily. 
