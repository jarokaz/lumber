# Lumber classification

http://www.ee.oulu.fi/research/imag/wood/WOOD/README

http://www.ee.oulu.fi/research/imag/wood/WOOD/


## Create a storage bucket for SaveModel
```
gsutil mb -l us-central1 gs://mlsandbox-staging
```


## Starting a local training job
```
gcloud ml-engine local train --module-name trainer.task --package-path trainer --job-dir ../jobdir -- --model vgg16base1 --tr
aining-file ../data/tfrecords/training.tfrecords --validation-file ../data/tfrecords/validation.tfrecords --hidden-units 512 --max-steps 10000 --eval-steps 1000
```

## Examine the exported model's signature
```
saved_model_cli show --dir current_savemodel --tag serve --signature_def serving_default
```

## Start a training job on ML-Engine
$JOB_NAME = vgg16base1_single_1
$




## Checking local prediction with glcoud

```
gcloud ml-engine local predict --model-dir current_savemodel --json-instances json_instances/image1.json
```

## Create 'bclassifier' model on CMLE Online Prediction

```
gcloud ml-engine models create bclassifier --regions us-central1
```

## List my models

```
gcloud ml-engine models list
gcloud ml-engine versions list --model bclassifier
```

## Deploy the model version
```
MODEL_BINARY=gs://mlsandbox-staging/vgg16base1v1
gcloud ml-engine versions create v1 --model bclassifier --origin $MODEL_BINARY --runtime-version 1.6
```
