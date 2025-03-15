## PHPRBP

Code and Datasets for "Phage Host Prediction Using Deep Neural Network with Multi-source Protein Language Models and Squeeze-and-Excitation Attention Mechanism"

### Datasets

See data folder


### Environment Requirement

Detailed package information can be found in PHPRBP.yaml.

Note: we suggest you to install all the package using conda (both miniconda and Anaconda are ok). After cloning this respository, you can use anaconda to install the PHPRBP.yaml. This will install all packages you need with cpu mode.

### Conducting phage host prediction

Users need to download the pre-training weights for ESM2 and ProtT5 before using PHPRBP. Place the weights under the pretrain_model folder.

ProtT5 (https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main)
ESM2 (https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main)

```
cd PHPRBP/code
```
```
python python pred.py --gpu 0 --dataset_name "Dataset1" --host_label "Host Genus" --batch_size 32 -- reduction 16 --drop_prob 0.4 --epochs 400
```
```
python python pred.py --gpu 0 --dataset_name "Dataset2" --host_label "Host Genus" --batch_size 32 -- reduction 16 --drop_prob 0.4 --epochs 400
```
```
python python pred.py --gpu 0 --dataset_name "Dataset3" --host_label "Host Species" --batch_size 32 -- reduction 16 --drop_prob 0.4 --epochs 200
```

