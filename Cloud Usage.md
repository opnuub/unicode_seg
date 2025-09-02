To bind the repository to Google Cloud Platform, follow the steps below:

1. In Google Cloud API & Services, enable 
    - Secret Manager API
    - Artifact Registry API
    - Vertex AI API
    - Cloud Pub/Sub API
    - Cloud Build API

2. Create a Google Cloud Storage Bucket, upload the dataset into gs://bucket_name/Data/ with the following directory structure:
Data/
├── Best/
│   ├── article/
│   ├── encyclopedia/
│   ├── news/
│   └── novel/
├── my_test_segmented.txt
├── my_train.txt
└── my_valid.txt

3. In Artifact Registry, create repository in the same region as the storage bucket. 

4. In Cloud Build, create a trigger in the same region as the Artifact Registry. 
    - Choose a suitable event (e.g. Push to a branch)
    - Select 2nd gen repository generation
    - Link the GitHub repository
    - Select Dockerfile (for Configurations) and Repository (for Location)
    - Dockerfile name: ```Dockerfile```, image name: ```us-central1-docker.pkg.dev/project-name/registry-name/image:latest```
    - Enable "Require approval before build executes"
    - For manual image build, press Enable/ Run in the created trigger

5. After image is created and stored in Artifact Registry, select "Train new model" under the Training tab in Vertex AI.
    - Training method: default (Custom training) and continue
    - Model details: fill in name and continue
    - Training container: select custom container and browse for latest built image, link to storage bucket and under arguments, modify and paste the following
    ```
    --path=gs://bucket_name/Data/
    --language=Thai
    --input-type=BEST
    --model-type=cnn
    --epochs=200
    --filters=32
    --name=Thai_codepoints_32
    --edim=40
    --embedding=codepoints
    ```
    - Hyperparameters: unselect and continue
    - Compute and pricing: choose existing resources or deploy to new worker pool
    - Prediction container: no prediction container and start training