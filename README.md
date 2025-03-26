<div align="center">
<h1> HHI-Assist <br> A Dataset and Benchmark of Human-Human Interaction in Physical Assistance Scenario </h1>
<h3>Saeed Saadatnejad, Reyhaneh Hosseininejad, Jose Barreiros, Katherine M. Tsui and Alexandre Alahi
</h3>
<h4> <i> under review, 2024 </i></h4>

 
[[webpage](https://sites.google.com/view/hhi-assist/home)]

<image src="docs/hhi2.jpg" width="600">
</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">
The increasing labor shortage and aging population underscore the need for assistive robots to support human care recipients. Developing such robots requires accurate human motion prediction to ensure their responsiveness and safety. This task is challenging due to the variability in scenarios and the necessity of modeling interactions between agents. To address these challenges, we first present HHI-Assist, a collection of motion capture clips of human-human interaction (HHI) for physical assistance. Second, we propose a conditional diffusion-based Transformer model that predicts the poses of interacting agents by effectively attending to the coupled relationships between caregivers and care receivers. Our approach demonstrates significant improvements over basic baselines and the generalization ability to new unseen demonstrations.
</br>
</br>

# Getting started

## Requirements
The code requires Python 3.8 or later. The file [requirements.txt](requirements.txt) contains the full list of required Python modules.
```
pip install -r requirements.txt
```
## Data directory
Data can be downloaded from [here](https://huggingface.co/datasets/jose-barreiros-tri/hhi-assist).

## Arguments explanation

### ```--data-dir```
This folder contains the HHI Dataset sequences.

### ```--epochs```
Number of epochs

### ```--input_n```
Number of frames the model takes as input

### ```--output_n```
Number of frames the model outputs

### ```--mode```
Choices: 
- train, to launch a training, model will select train split of the data directory you give
- test, to launch an evaluation, model will select the test split of the data directory you give

### ```--output_dir```
When mode = train : the output directory where your model will be saved \
When mode = test : the model's path you want to run an evaluation test on

### ```--resume```
Resume training of a finished training model. If for example you have a model that was trained for 20 epochs, and you want to resume training it for 5 more epochs, you add ```--resume --epochs 25``` 
to your command line.
 
### ```--joints```
The number of joints of the human pose you are trying to predict. **Default value is 20**

### ```--name```
The name of csv results file where your results (mpjpe per timestep) is stored.

### ```--layers```
Number of transformer layers in the architecture.

### ```--lr```
Learning rate. Default value: 1.0e-3

### ```--no_hip```
Add flag to not apply hip translation. **Default is : False.**, i.e. hip translation is applied by default.
Applies only to single sequence input case.
For models that predicts two human poses, different ```--h``` flag choices takes care of this.

### ```--angles```
Choices: 
- 'None', is **default value**, set to use joints' xyz coordinates
- 'Quat', set to use joints' angles information in quaternions representation.
- 'RotMatrix', set to use joints' angles information in rotation matrices representation.

### ```--reload_dataset```
**Default value** is True. 
Code by defaults caches the last dataset used.
Set to False if you do not want to wait for program to parses and loads all files again, and use last cached dataset.

### ```--num_steps```
Number of sampling steps the denoiser performs when generating samples.

### ```--h```
This parameter defines how dataset pair is loaded:

0 : hip translation, no hip joint
1 : no hip translation, with hip joint
2 : hip translation, with hip joint
3 : hip translation, with difference of hip joints

### ```--shift```
To do the CG or CR shifted half a second in the future experiment, 
Choices:
- CR : CR sequence will be shifted 0.5 seconds in to the future, CG sequence last 0.5 seconds will be cutoff
- CG : CG sequence will be shifted 0.5 seconds in to the future, CR sequence last 0.5 seconds will be cutoff

### ```--batch```
The batch size.


### ```--generalization_experiment ```

Setting this flag will select the Generalization Experiment data split.

### ```--baselines```

If you want to test other baselines, indicate their names.


# Training and Testing

The model trained with the following command:\
```python main.py  --data-dir PATH_TO_DATA  --output_dir PATH_TP_OUTPUT --joints 21 --epochs 20 --batch 64```.

To evaluate it on the test set, run: \
```python main.py  --data-dir PATH_TO_DATA  --output_dir PATH_TP_OUTPUT --joints 21 --epochs 20 --batch 64 --mode test ```\

To get resuls of other baselines (ConstVel, ZeroVel, etc), please refer to ```--baselines```.