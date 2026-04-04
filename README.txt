This bundle is for deploying the persistent CSV pole inference server to Paperspace Deployments.

Files:
- Dockerfile
- requirements.txt
- deployment.yaml
- pole_csv_persistent_server.py  (copy into this folder before build)
- best_model.pt                  (copy/rename your trained model into this folder before build)

Build + push:
  docker login -u YOUR_DOCKERHUB_USERNAME
  docker build -t pole-csv-persistent .
  docker tag pole-csv-persistent YOUR_DOCKERHUB_USERNAME/pole-csv-persistent:latest
  docker push YOUR_DOCKERHUB_USERNAME/pole-csv-persistent:latest
