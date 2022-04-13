# mnistSigOpt
## SigOpt can be run in two modes:​

https://app.sigopt.com/docs/tutorial/experiment ​

### Runs: logging of individual runs​
    sigopt run python <mnist_train.py> <ArgParse parameters>
### Optimization: Tuning hyperparameters of the model​
    sigopt optimize -e <experiment.yaml> python <mnist_train.py> <ArgParse parameters>​
