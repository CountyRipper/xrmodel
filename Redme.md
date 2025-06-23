## xrmodel 
KG/LG的第二进阶
包含LLM，prefixLM，新款t2t model的尝试


关于verbalizer/ranking，也考虑使用新版的cross-encoder/bi-encoder
负采样方面，尝试新方式

### Label Generation
Using transformer-encoder&decoder structure model such as T5, BART to generate labels

The model file is `gen_model.py`

jupyter notebook: `gen_model.ipynb`
