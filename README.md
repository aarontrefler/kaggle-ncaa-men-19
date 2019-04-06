2019 NCAA Men's Tournament Kaggle Competition
==============================
Aaron Trefler  
Project Duration: Part-time work from Feb 25th to March 21st 2019

Introduction
------------
This project was created in order to compete in Kaggle's 2019 ML NCAA Mens Competition.

The competition required submission of winning probabilities for every potential match-up of the 2019 tournament (i.e., 68teams^2 * 68teams^2 / 2 = 2,312 match-ups).

The competition is scored by assessing the negative log loss score of your predictions. Lower scores are better.

Executing Project
------------
In order to run the entire project (i.e., create all datasets and models):
1. ensure this projects `src` directory is put on your `$PYTHONPATH`, by altering `SRC_PATH` variable in `src:run.sh`
2. alter the `project_path` variable in `src:utils.py` to that of this project's directory on your machine
3. `cd` into the `src` directory and execute `sh run.sh`.

Project assumes you are running Python 3, and have all necessary packages.

Data Files
------------
Data files used in this project were provided by Kaggle. Specifically the following raw files are used:
- `Teams.csv`: team names matched to team IDs
- `NCAATourneySeeds.csv`: tournament seedings
- `RegularSeasonDetailedResults.csv`: regular season results from 2003-2019
- `NCAATourneyCompactResults`: tournament results from 1985-2019
- `SampleSubmissionStage2.csv`: formatted Kaggle submission file for 2019 tournament

Features Used
------------
108 features were used. They can be grouped in the following categories:
- Team Seeds
- Average Regular Season Statistics (e.g., assists, blocks, scores)
- Average Regular Season Statistics of Opponents
- Advanced Regular Season Statistics (e.g., net rating, offensive rating, pace)
- Match-up Differences (e.g., seed difference, net rating difference)

Modeling
------------
Random Forest classifier was chosen.

Cross validation split was performed by creating training (1985-2013 games) and validation (2014-2019 games) datasets.

Hyper-parameter optimization was performed by feeding the training and validation datasets into a grid-search over manually set hyper-parameter values.

Final model used to make 2019 predictions used the grid-search optimal hyper-parameters and was trained on the entire modeling dataset (i.e., training and validation sets).

Jupyter (iPython) Notebooks
------------
Notebooks can only be executed after project has been run.

Notebooks in this project are as follows:
- `AT-1.1-analyze-dataset`: Analyze and view created datasets
- `AT-1.1-analyze-model`: Analyze trained model and cross valid results

Project Organization
------------
    ├── README.md
    ├── data
    │   ├── clean                      <- Features datasets
    │   ├── interim                    <- Processed datasets
    │   └── raw                        <- Data files downloaded from Kaggle
    │
    ├── models                          
    │   ├── clf.model                  <- Model object trained using training dataset
    │   ├── gridcv.model               <- GridSearch object containing trained model using entire dataset
    │   ├── reports                    <- Training and validation dataset scores
    │   └── submissions                <- Kaggle submission file
    │
    ├── src
    │   ├── run.sh                     <- Execute entire project 
    │   ├── utils.sh                   <- Project level utility functions
    │   ├── data                       
    │   │   ├── make_dataset.py        <- Create datasets in interim directory
    │   │   └── data_utils.py
    │   │ 
    │   ├── features
    │   │   ├── build_features.py      <- Create datasets in `clean` directory
    │   │   └── feature_utils.py 
    │   │  
    │   └── models                    
    │       ├── train_model.py         <- Train models and make predictions using modeling data
    │       ├── predict_model.py       <- Make predictions on 2019 games
    │       └── model_utils.py         
    │
    └── notebooks                      <- Jupyter notebooks

