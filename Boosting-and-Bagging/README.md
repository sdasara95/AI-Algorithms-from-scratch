# Boosting and Bagging

Implemented K-Nearest Neighbors, Adaboost and Decision Forest from scratch for image orientation classification.<br>
The input is a compressed vector from image data and the label is image orientation angle.
Kindly check the the pdf files for detailed report about performance of each model.
Our best model is KNN so, the output of KNN and best.txt is same. So, we uploaded only KNN output(no best output file).

Run the following commands:

Training:
```
./orient.py train train-data.txt nearest_model.txt nearest

./orient.py train train-data.txt adaboost_model.txt adaboost

./orient.py train train-data.txt forest_model.txt forest

./orient.py train train-data.txt best_model.txt best
```

Testing
```
./orient.py test test-data.txt nearest_model.txt nearest

./orient.py test test-data.txt adaboost_model.txt adaboost

./orient.py test test-data.txt forest_model.txt forest

./orient.py test test-data.txt best_model.txt best
```

Output predictions are stored in the following files:
```
output_nearest.txt

output_adaboost.txt

output_forest.txt
```
