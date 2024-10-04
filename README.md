# MotionCraft: Physics-Based Zero-Shot Video Generation

Code for paper "MotionCraft: Physics-Based Zero-Shot Video Generation" by Luca Savant Aira, Antonio Montanaro, Emanuele Aiello, Diego Valsesia, Enrico Magli.

BibTex references for journal and conference versions:

```
@inproceedings{savantaira2024motioncraft,
      title={MotionCraft: Physics-based Zero-Shot Video Generation},
      author={Savant Aira, Luca and Montanaro, Antonio and Aiello, Emanuele and Valsesia, Diego and Magli, Enrico},
      booktitle={Advances in Neural Information Processing Systems},
      year={2024}
}
```

## News
* [25/09/2024] MotionCraft got accepted ad Neurips2024! 🎉️
* [04/10/2024] First version of the code released!

## Setup and Usage


1. Clone this repository and enter:

``` shell
git clone https://github.com/mezzelfo/MotionCraft.git
cd MotionCraft/
```
2. Install the required packages and activate the environment:
``` shell
conda env create -f environment.yml
conda activate MotionCraft
```
3. Run MotionCraft on the provided examples:
``` shell
bash generate_videos.sh
```

## TODO

- [ ] Add more examples
- [ ] Add script for the flow correlation experiment
- [ ] Add script for the ablation study
- [ ] Add script for computing the quantitative metrics

