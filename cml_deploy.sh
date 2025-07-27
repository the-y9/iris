#!/bin/bash
set -e

echo "🔧 Building Docker image..."
docker build -t iris-api .

echo "🔧 Tagging Docker image..."
docker tag iris-api us-central1-docker.pkg.dev/eternal-cycling-463617-f7/iris-repo/iris-api:latest

echo "📤 Pushing Docker image to Artifact Registry..."
docker push us-central1-docker.pkg.dev/eternal-cycling-463617-f7/iris-repo/iris-api:latest

echo "🚀 Deploying to GKE..."
kubectl apply -f k8s-deployment.yaml

echo "✅ Deployment complete."
