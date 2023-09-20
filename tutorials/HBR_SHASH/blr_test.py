
import pcntoolkit as ptk
import os
import pandas as pd

inputdir = os.getcwd()
processing_dir = '/Users/johannabayer/Documents/Github/PCNtoolkit-demo/tutorials/HBR_SHASH/HBR_tutorial_wdir'
model_dir = '/Users/johannabayer/Documents/Github/PCNtoolkit-demo/tutorials/HBR_SHASH/HBR_tutorial_wdir'

respfile = os.path.join(processing_dir, 'Y_train.pkl')       # measurements  (eg cortical thickness) of the training samples (columns: the various features/ROIs, rows: observations or subjects)
covfile = os.path.join(processing_dir, 'X_train.pkl')        # covariates (eg age) the training samples (columns: covariates, rows: observations or subjects)

testrespfile_path = os.path.join(processing_dir, 'Y_test.pkl')       # measurements  for the testing samples
testcovfile_path = os.path.join(processing_dir, 'X_test.pkl')        # covariate file for the testing samples

def ldpkl(filename: str): 
    with open(filename, 'rb') as f:
        return pickle.load(f)

ptk.normative.estimate(respfile=os.path.join(processing_dir,'Y_train.pkl'),
    covfile=os.path.join(processing_dir,"X_train.pkl"),
    testresp=os.path.join(processing_dir,"Y_test.pkl"),
    testcov=os.path.join(processing_dir,"X_test.pkl"),
    cvfolds=2,
    alg="blr",
    optimizer = "powell",
    outputsuffix= "_2fold",
    saveoutput=True,
    savemodel=True,
    standardize = True)

covfile=os.path.join(processing_dir,"X_pred.pkl")


print(os.path.join(processing_dir,'X_train.pkl'))
ptk.normative.predict(covfile=os.path.join(processing_dir,'X_train.pkl'),
  respfile=os.path.join(processing_dir,'Y_train.pkl'),
  model_path=os.path.join(processing_dir,"Models"),
  alg="blr",
  outputsuffix="_test",
  return_y=True
)