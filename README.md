<!-- <center>![RefineCTR](https://github.com/codectr/RefineCTR/blob/main/RefineCTR.png)<center> -->
<div align="center"><img src="https://github.com/codectr/RefineCTR/blob/main/RefineCTR.png"></div>
The code and experimental results of《A Comprehensive Analysis and Evaluation of Feature Refinement Modules for CTR Prediction》.  

Click-through rate (CTR) prediction is widely used in academia and industry. Most CTR prediction models follow the same design paradigm: Feature Embedding (FE) layer, Feature Interaction (FI) layer, and Prediction layer. RefineCTR inserts existing feature refinement modules between FE and FI layer to boost basic CTR models performance, which also sumarizes and compares those modules' effectiveness comprehensively.  The structure of RefinedCTR is shown in the following:

<div align="center"><img src="https://github.com/codectr/RefineCTR/blob/main/evaluation/figure/refinectr%20structure.png" width="40%"></div>

The implement of FR modules is shown in the following figure : (A) and (B) inserting a single FR module between FR and FI layer for stacked and parallel CTR models; (C) assigning two separate FR modules for different FI sub-network to generate discriminate feature distributions for parallel models.



<!-- <center>![The primary backbone structures of common CTR prediction models ](https://github.com/codectr/RefineCTR/blob/main/refineCTR%20framework.png)<center> -->
<div align="center" size=><img src="https://github.com/codectr/RefineCTR/blob/main/refineCTR%20framework.png" width="60%"></div>


# Feature Refinement Modules

We extrct 14 FR modules from existing works. And FAL equires that the input data not anonymous. Therefore, we do not evaluate FAL in the experiments as Criteo dataset is anonymous. It is worth noting that we would fix the hyper-parameters of the basic CTR models when applying these modules to ensure the fairness of the experiments. Relatively speaking, this also leaves space for improvement for each augmented model.

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
| 16   | AFM           | IJCAI'17    | A   SIngle   |
| 17   | NFM           | SIGIR'17    | A   SIngle   |
| 18   | FwFM          | WWW'18      | A   SIngle   |
| 19   | FINT          | arXiv' 21   | A   SIngle   |
| 20   | PNN           | ICDM'16     | A   SIngle   |
| 21   | FiBiNET       | RecSys'19   | C   Separate |
| 22   | DCAP          | CIKM'21     | A   SIngle   |
| 23   | AutoInt       | CIKM'19      | A   SIngle   |
| 24   | AutoIntP      | CIKM'19     | B   SIngle   |
| 25   | AutoIntP      | CIKM'19      | C   Separate |

Generally, stacked models (e.g., FM. CN) only use pattern A (Single FR module); parallel models can adopt both pattern B (single FR module) and Pattern C (two separate FR modules).
Including basic model, we can generate N*14(N is the basic model number) augmented  models. Meanwhile, for parallel CTR model,  we can assigning different FR module for different FI sub-networks. Now we assign two same FR modules for different sub-networks.  

# Experiment Results

We will continue to update and upload the latest experimental results and analysis.


## Criteo
### AUC of Criteo
| Modules   | SKIP   | FEN    | SENET  | FWN    | DFEN   | DRM    | VGate  | BGate  | SelfAtt | TCE    | PFFN   | GFRL   | FRNet-V | FRNet-B |
| --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- | ------ | ------ | ------ | ------- | ------- |
| FM        | 0.8080 | 0.8100 | 0.8102 | 0.8100 | 0.8117 | 0.8107 | 0.8090 | 0.8091 | 0.8099  | 0.8112 | 0.8129 | 0.8134 | 0.8139  | 0.8140  |
| DeepFM    | 0.8121 | 0.8128 | 0.8125 | 0.8125 | 0.8121 | 0.8118 | 0.8125 | 0.8127 | 0.8112  | 0.8123 | 0.8129 | 0.8137 | 0.8140  | 0.8141  |
| DeepFM(2) | 0.8121 | 0.8130 | 0.8128 | 0.8129 | 0.8123 | 0.8119 | 0.8128 | 0.8131 | 0.8129  | 0.8128 | 0.8132 | 0.8138 | 0.8142  | 0.8142  |
| CN        | 0.8093 | 0.8102 | 0.8095 | 0.8094 | 0.8121 | 0.8109 | 0.8107 | 0.8110 | 0.8102  | 0.8122 | 0.8130 | 0.8139 | 0.8143  | 0.8144  |
| DCN       | 0.8125 | 0.8130 | 0.8116 | 0.8127 | 0.8127 | 0.8118 | 0.8124 | 0.8127 | 0.8122  | 0.8126 | 0.8131 | 0.8142 | 0.8143  | 0.8145  |
| DCN(2)    | 0.8125 | 0.8136 | 0.8122 | 0.8126 | 0.8131 | 0.8120 | 0.8124 | 0.8127 | 0.8129  | 0.8133 | 0.8132 | 0.8144 | 0.8144  | 0.8146  |
| AFN       | 0.8099 | 0.8140 | 0.8104 | 0.8106 | 0.8116 | 0.8110 | 0.8103 | 0.8101 | 0.8110  | 0.8122 | 0.8130 | 0.8130 | 0.8139  | 0.8141  |
| AFN+      | 0.8108 | 0.8141 | 0.8111 | 0.8124 | 0.8119 | 0.8125 | 0.8119 | 0.8118 | 0.8129  | 0.8128 | 0.8132 | 0.8141 | 0.8141  | 0.8141  |
| AFN+(2)   | 0.8108 | 0.8142 | 0.8116 | 0.8128 | 0.8127 | 0.8129 | 0.8124 | 0.8126 | 0.8131  | 0.8131 | 0.8134 | 0.8142 | 0.8143  | 0.8145  |
| CN2       | 0.8121 | 0.8119 | 0.8119 | 0.8140 | 0.8131 | 0.8140 | 0.8128 | 0.8133 | 0.8131  | 0.8138 | 0.8130 | 0.8143 | 0.8141  | 0.8143  |
| DCNV2     | 0.8128 | 0.8129 | 0.8122 | 0.8142 | 0.8134 | 0.8141 | 0.8130 | 0.8138 | 0.8135  | 0.8136 | 0.8130 | 0.8143 | 0.8141  | 0.8143  |
| DCNV2(2)  | 0.8128 | 0.8136 | 0.8125 | 0.8140 | 0.8135 | 0.8142 | 0.8139 | 0.8140 | 0.8137  | 0.8138 | 0.8131 | 0.8144 | 0.8143  | 0.8144  |
| IMP       | -      | 0.18%  | 0.03%  | 0.13%  | 0.15%  | 0.12%  | 0.09%  | 0.11%  | 0.11%   | 0.18%  | 0.22%  | 0.33%  | 0.35%   | 0.37%   |

### Logloss of Criteo
| Modules   | SKIP   | FEN    | SENET  | FWN    | DFEN   | DRM    | VGate  | BGate  | SelfAtt | TCE    | PFFN   | GFRL   | FRNet-V | FRNet-B |
| --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- | ------ | ------ | ------ | ------- | ------- |
| FM        | 0.4437 | 0.4418 | 0.4416 | 0.4418 | 0.4401 | 0.4411 | 0.4425 | 0.4424 | 0.4408  | 0.4405 | 0.4391 | 0.4384 | 0.4380  | 0.4378  |
| DeepFM    | 0.4398 | 0.4390 | 0.4413 | 0.4393 | 0.4399 | 0.4400 | 0.4392 | 0.4390 | 0.4404  | 0.4396 | 0.4390 | 0.4382 | 0.4379  | 0.4378  |
| DeepFM(2) | 0.4398 | 0.4389 | 0.4410 | 0.4389 | 0.4396 | 0.4400 | 0.4390 | 0.4386 | 0.4389  | 0.4390 | 0.4387 | 0.4380 | 0.4378  | 0.4377  |
| CN        | 0.4498 | 0.4415 | 0.4432 | 0.4496 | 0.4398 | 0.4409 | 0.4409 | 0.4406 | 0.4417  | 0.4398 | 0.4392 | 0.4380 | 0.4378  | 0.4375  |
| DCN       | 0.4394 | 0.4389 | 0.4419 | 0.4391 | 0.4393 | 0.4400 | 0.4394 | 0.4391 | 0.4395  | 0.4396 | 0.4392 | 0.4377 | 0.4376  | 0.4375  |
| DCN(2)    | 0.4394 | 0.4383 | 0.4405 | 0.4391 | 0.4389 | 0.4399 | 0.4393 | 0.4391 | 0.4389  | 0.4386 | 0.4388 | 0.4376 | 0.4374  | 0.4374  |
| AFN       | 0.4420 | 0.4378 | 0.4412 | 0.4411 | 0.4402 | 0.4409 | 0.4413 | 0.4416 | 0.4408  | 0.4396 | 0.4391 | 0.4388 | 0.4379  | 0.4377  |
| AFN+      | 0.4410 | 0.4378 | 0.4407 | 0.4394 | 0.4399 | 0.4394 | 0.4398 | 0.4399 | 0.4389  | 0.4390 | 0.4390 | 0.4378 | 0.4378  | 0.4378  |
| AFN+(2)   | 0.4410 | 0.4378 | 0.4401 | 0.4390 | 0.4392 | 0.4388 | 0.4393 | 0.4391 | 0.4388  | 0.4388 | 0.4388 | 0.4377 | 0.4376  | 0.4374  |
| CN2       | 0.4389 | 0.4402 | 0.4402 | 0.4380 | 0.4393 | 0.4381 | 0.4394 | 0.4386 | 0.4387  | 0.4383 | 0.4393 | 0.4378 | 0.4379  | 0.4378  |
| DCNV2     | 0.4390 | 0.4391 | 0.4399 | 0.4379 | 0.4389 | 0.4380 | 0.4390 | 0.4382 | 0.4383  | 0.4385 | 0.4391 | 0.4378 | 0.4378  | 0.4378  |
| DCNV2(2)  | 0.4390 | 0.4382 | 0.4394 | 0.4380 | 0.4388 | 0.4379 | 0.4380 | 0.4380 | 0.4383  | 0.4383 | 0.4391 | 0.4376 | 0.4378  | 0.4377  |
| IMP       | -      | 0.44%  | 0.03%  | 0.22%  | 0.35%  | 0.33%  | 0.29%  | 0.35%  | 0.35%   | 0.43%  | 0.45%  | 0.70%  | 0.74%   | 0.77%   |

### Assigning Two Seperate Feature Refinement Modules for Different sub-networks based on DeepFM.

<table border=0,rules=none><tr> <td><img src="https://github.com/codectr/RefineCTR/blob/main/evaluation/figure/deepfm_auc.jpg" border=0></td> <td><img src="https://github.com/codectr/RefineCTR/blob/main/evaluation/figure/deepfm_ll.jpg" border=0></td> </tr></table> 

## Frappe
### AUC of Frappe
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


| Modules    | SKIP   | FEN    | SENET  | FWN    | DFEN   | DRM    | VGate  | BGate  | SelfAtt | TCE    | PFFN   | GFRL   | FRNet-V | FRNet-B |
| ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- | ------ | ------ | ------ | ------- | ------- |
| CIN        | 0.9834 | 0.9839 | 0.9835 | 0.9830 | 0.9837 | 0.9836 | 0.9772 | 0.9787 | 0.9813  | 0.9792 | 0.9837 | 0.9837 | 0.9845  | 0.9842  |
| xDeepFM    | 0.9841 | 0.9846 | 0.9844 | 0.9836 | 0.9836 | 0.9846 | 0.9844 | 0.9835 | 0.9830  | 0.9839 | 0.9837 | 0.9850 | 0.9851  | 0.9851  |
| xDeepFM(2) | 0.9841 | 0.9841 | 0.9844 | 0.9840 | 0.9841 | 0.9847 | 0.9850 | 0.9845 | 0.9832  | 0.9833 | 0.9831 | 0.9852 | 0.9853  | 0.9854  |
| NFM        | 0.9818 | 0.9830 | 0.9816 | 0.9799 | 0.9830 | 0.9825 | 0.9845 | 0.9830 | 0.9808  | 0.9818 | 0.9828 | 0.9839 | 0.9843  | 0.9855  |
| AFM        | 0.9697 | 0.9814 | 0.9797 | 0.9804 | 0.9808 | 0.9771 | 0.9718 | 0.9724 | 0.9793  | 0.9776 | 0.9801 | 0.9812 | 0.9816  | 0.9830  |
| FwFM       | 0.9834 | 0.9809 | 0.9834 | 0.9813 | 0.9834 | 0.9816 | 0.9830 | 0.9819 | 0.9840  | 0.9834 | 0.9823 | 0.9842 | 0.9831  | 0.9837  |
| FINT       | 0.9832 | 0.9845 | 0.9817 | 0.9828 | 0.9828 | 0.9842 | 0.9838 | 0.9833 | 0.9825  | 0.9822 | 0.9827 | 0.9848 | 0.9834  | 0.9840  |
| PNN        | 0.9841 | 0.9843 | 0.9846 | 0.9837 | 0.9836 | 0.9845 | 0.9835 | 0.9836 | 0.9828  | 0.9833 | 0.9831 | 0.9837 | 0.9842  | 0.9843  |
| FiBiNet    | 0.9827 | 0.9825 | 0.9825 | 0.9827 | 0.9829 | 0.9824 | 0.9828 | 0.9828 | 0.9830  | 0.9823 | 0.9819 | 0.9832 | 0.9835  | 0.9846  |
| DCAP       | 0.9841 | 0.9836 | 0.9845 | 0.9834 | 0.9843 | 0.9857 | 0.9846 | 0.9838 | 0.9823  | 0.9841 | 0.9824 | 0.9847 | 0.9853  | 0.9848  |

# Get started

1. **Test existing model with existing  module**.
Users can choose the appropriate CTR model and feature refinement module conveniently according their needs.
```
cd evaluation/mains
CUDA_VISIBLE_DEVICES=0 python main_frappe_base --model 0 --module 0
```

2. **Adding new model or module.**

   Our framework RefineCTR is modularized, users can adjust or add basic models and modules easily. 
