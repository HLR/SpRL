# Spatial Language Understanding
This project contains implementation codes of paper [1].

## Configurations
All configurations needed to run this application are placed in 
[`pairConfigurator`](pairConfigurator.scala). In order to run the application for training, set `IsTraining` to `Configurator.TRUE` and for testing set it to `Configurator.FALSE`. 


## Running
To run the main app with default properties:

```
sbt "project saulExamples" "runMain edu.tulane.cs.hetml.nlp.sprl.Pairs.MultiModalPairSpRLApp"
```

## References
[1] Parisa Kordjamshidi, Taher Rahgooy, and Umar Manzoor. "Spatial Language Understanding with Multimodal Graphs using Declarative Learning based Programming." Proceedings of the 2nd Workshop on Structured Prediction for Natural Language Processing. 2017.