# Iris Classifier API - File Overview

- `.github/workflows/deploy.yml`  
  Continuous Deployment pipeline using GitHub Actions and CML.

- `Dockerfile`  
  Defines Docker image build steps for the Iris API.

- `cml_deploy.sh`  
  Script to build, tag, push Docker image and deploy to Kubernetes.

- `iris_fastapi.py`  
  FastAPI application code serving the Iris classifier API.

- `k8s-deployment.yaml`  
  Kubernetes manifest for deploying the API and exposing the service.

- `req.txt`  
  Python dependencies for the API.

- `week6_docker.txt`  
  Shell command history for creating and runing docker images.
