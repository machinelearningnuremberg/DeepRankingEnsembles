# Deep Ranking Ensembles
Repository for Deep Ranking Ensembles for Hyperparameter Optimization (ICLR 2023 Conference Paper942).

# Setup
## Environment Setup
* Install anaconda/miniconda according to the [Installation Instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html "Named link title").
* Clone this repository into a folder. `git clone --recurse-submodules <url> ./repo`
* Go inside the `repo` folder. `cd repo`
* Create the required conda environment. `conda env create --file linux_environment.yml`
* If the environment is already created, update it using the command `conda env update --file linux_environment.yml --prune`.
* This creates a conda environment called `DRE`. Activate the conda environment using command `conda activate DRE`.
* After activating `DRE` environment, display the script usage help message using command `python DRE.py -h`.
* Please download the data as described in the next section before running the `DRE.py` script.

## Data Download
* Download the HPO\_B data from [HERE](https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip). A file named `hpob-data.zip` will be downloaded.
* Extract `hpob-data.zip` to the location `./repo/HPO_B/hpob-data/`. Here the `./repo/HPO_B` folder is the location where the HPO\_B submodule is cloned.
* After successful extraction, all the required files (in the json format) will be present in the `./repo/HPO_B/hpob-data/` folder.
* The extracted files should be in the following folder structure:
    * `repo`
        * `HPO_B`
            * `hpob-data`
                * `bo-initializations.json`
                * `meta-test-dataset.json`
                * `meta-test-tasks-per-space.json` (This file is present in the folder by default)
                * `meta-train-dataset-augmented.json`
                * `meta-train-dataset.json`
                * `meta-validation-dataset.json`

## Example command
* To train & evaluate the DeepRankingSurrogate, you can run the following commands chronologically.
    * `python DRE.py --train --train_index 0 --meta_features --M 10 --layers 4 --result_folder ./results_M10/`. After running this,
       a file named '4796' which is the search space ID corresponding to index 0 is created in the `./results_M10` folder. Please check
       the next section.
    * `python DRE.py --evaluate --eval_index 0 --meta_features --M 10 --layers 4 --result_folder ./results_M10/`. After running this,
       a file named `EVAL_KEY_0` will be created in the `./results_M10` folder.
* Please make sure that the architecture and the result folder for the training and evaluation is identical.


## Index Table
The following table shows the `train_index` and `eval_index` corresponding to each search space in the hpob\_data. Each `train_index` refers to a single Search Space ID. On the other hand, each Search Space has a range of evaluation indices. The set of all evaluation indices of a search space is given by the cross product: {Set containing the IDs of all Datasets of the search space} X {Set containing random seeds used to start the BO iteration}. For example `eval_index = 0` corresponds to `('4796', '23','test0')`.

| Search Space ID | Train Index | Eval Index  |
| :-----------:    | :---------: | :---------: |
|     4796         |      0      |  0   - 19   |
|     5527         |      1      |  20  - 49   | 
|     5636         |      2      |  50  - 79   |
|     5859         |      3      |  80  - 109  |
|     5860         |      4      |  110 - 124  |
|     5889         |      5      |  125 - 134  |
|     5891         |      6      |  135 - 164  |
|     5906         |      7      |  165 - 174  |
|     5965         |      8      |  175 - 209  |
|     5970         |      9      |  210 - 239  |
|     5971         |      10     |  240 - 269  |
|     6766         |      11     |  270 - 299  |
|     6767         |      12     |  300 - 329  |
|     6794         |      13     |  330 - 359  |
|     7607         |      14     |  360 - 394  |
|     7609         |      15     |  395 - 429  |


