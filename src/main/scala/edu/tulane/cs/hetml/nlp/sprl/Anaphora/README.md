# Anaphora Resolution
In this project we include the implementation codes of [1] paper.

## Dataset
Please download the dataset from here (??) and place it in the root directory of the application. The layout of the data should be as follows:
```bash
Data/
|-- saiapr_tc-12
|   |-- VGData
|   |   |-- image_data.json  
|   |   |-- relationships.json
|   `-- tripletHeadWords
|       |-- tripletsHeadWordsTest.txt  
|       |-- tripletsHeadWordsTrain.txt
|-- newSprl2017_train.xml
|-- newSprl2017_gold.xml
```
 
## AnaphoraApp
All configurations needed to run the main application are placed in 
[`tripletConfigurator`](tripletConfigurator.scala). In order to run the application for training, set `isTrain` to `true` and for testing set it to `false`. 


## Running

To run the main app with default properties:

```
sbt "project saulExamples" "runMain edu.tulane.cs.hetml.nlp.sprl.Triplets.CoReferenceTripletApp"
```

results will be saved in `data/mSprL/results` folder as text files corresponding to the model selected in the config file. 

## Results on CLEF 2017 dataset
Here are the summarized results of relation classifier for different models
```
label                           Precision  Recall     F1             
---------------------------------------------------------------         
M0                              65.64      60.23      62.82
M0+C                            70.04      66.55      68.25 
A-Replacement                   78.47      56.84      65.92
A-Inference                     70.23      68.25      69.23
```

## References
[1] Umar Manzoor, and Parisa Kordjamshidi. "??". In preparation