To bind the repository to Google Cloud Platform, follow the steps below:

1. In Google Cloud API & Services, enable 
    - Cloud Logging API
    - Artifact Registry API
    - Vertex AI API
    - Cloud Pub/Sub API
    - Cloud Build API

2. Create a Google Cloud Storage Bucket, upload the dataset into gs://bucket_name/Data/ with the following directory structure:\
Data/\
├── Best/\
│   ├── article/\
│   ├── encyclopedia/\
│   ├── news/\
│   └── novel/\
├── my_test_segmented.txt\
├── my_train.txt\
└── my_valid.txt

3. In Artifact Registry, create repository in the same region as the storage bucket. In cloudbuild.yaml, change the commented lines to the path of repository created + image name.
```bash
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: [
    'build',
#   '-t', 'europe-west2-docker.pkg.dev/crested-return-452614-t4/test/test:latest',
    '.'
  ]
  id: 'Test Image'

options:
 logging: CLOUD_LOGGING_ONLY

images:
# - 'europe-west2-docker.pkg.dev/crested-return-452614-t4/test/test:latest'
```

4. In Cloud Build, create a trigger in the same region as the Artifact Registry. 
    - Choose a suitable event (e.g. Push to a branch)
    - Select 2nd gen repository generation
    - Link the GitHub repository
    - Select Cloud Build configuration file (for Configurations), Repository (for Location)
    - Enable "Require approval before build executes"
    - For manual image build, press Enable/ Run in the created trigger

5. After image is created and stored in Artifact Registry, select "Train new model" under the Training tab in Vertex AI.
    - Training method: default (Custom training) and continue
    - Model details: fill in name and continue
    - Training container: select custom container and browse for latest built image, link to storage bucket and under arguments, modify and paste the following
    ```bash
    --path=gs://bucket_name/Data/
    --language=Thai 
    --input-type=unsegmented
    --epochs=5
    --name=test
    ```
    - Hyperparameters: unselect and continue
    - Compute and pricing: choose existing resources or deploy to new worker pool
    - Prediction container: no prediction container and start training