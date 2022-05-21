# MTKDSR

## Multi-Teacher Knowledge Distillation for Super Resolution Image Reconstruction 

## Command

```shell
python main.py --model_name S_Model --checkpoint None --train_by_TT True --t1_model_name T1 --t1_checkpoint T1_C.pth --t2_model_name T--t1_checkpoint T2_C.pth 
# you can set the setting in the main.py instead of in the shell command.

# test
python main.py --Train False
# you can set the test datasets in the main.py too.
```

## Method

### The traditional training methods of SISR(a and b). 

![Figure1](https://user-images.githubusercontent.com/37239596/169652716-e74f988e-f2b4-4d14-9317-882f56a5e92b.png)

### Our Method.
![Figure2](https://user-images.githubusercontent.com/37239596/169652762-e1ad5a84-ece4-40b4-babc-8ae2042e6131.png)

## Experiment Result.
![Figure3](https://user-images.githubusercontent.com/37239596/169652802-6b2df774-095d-4b1b-a742-8bbcb3b6841d.png)
