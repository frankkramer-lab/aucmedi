```bash
# Pull the image from the Container Registry
docker pull ghcr.io/frankkramer-lab/aucmedi

# Run a container call
docker run \
  -v /home/dominik/aucmedi.data:/data \
  --rm \
  ghcr.io/frankkramer-lab/aucmedi \
  training \
  --architecture "DenseNet121"
```



docker run ghcr.io/frankkramer-lab/aucmedi:latest training

Volumes:
If you intended to pass a host directory, use absolute path.!!!

Be aware of user changes due to docker volume
-> created files will have user and group root

docker run -v /home/mudomini/projects/testing_area/aucmedi.automl:/data --rm -it domi:latest prediction








docker run ghcr.io/frankkramer-lab/aucmedi:latest training

docker build -t ghcr.io/frankkramer-lab/aucmedi:latest -t ghcr.io/frankkramer-lab/aucmedi:v0.7.0 .
