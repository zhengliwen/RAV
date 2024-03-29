# RAV
Source code and dataset for RAV.

## Requirements
Run the command for running environment.

```
pip install -r requirements.txt
```

## Data Preparation
Go to RAV/samples to check the input data format.

Download the preprocessed data from  [[data_path](https://drive.google.com/drive/folders/19dT03A8hLakaUYttAaTgX4S-5tgmCilP)] , and put them to folder "RAV/data" 

### RAV/data - filetree
```
├── /RAV/
│  ├── /data/
│  │  ├── 30_doc_dev_9999.tsv
│  │  └── 30_doc_test_9999.tsv
│  │  └── 30_doc_train_14w.tsv
```

## GAV Training
Download model checkpoint from [[model_path](https://drive.google.com/drive/folders/18_CU8pKq0lAQy9AUYfakJrNbfqkBYVKE)] , or train the model by running this command.

### RAV/output - filetree
```
├── /RAV/
│  ├── /output/
│  │  ├── best.pth.tar
│  │  └── dev-results.tsv
│  │  └── results.txt
│  │  └── test-results.tsv
```

```
python RAV.py
```
## GAV Testing
```
python test.py
python results_scorer.py
```

## Citation

