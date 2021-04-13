## Docker Build

This Dockerfile builds a Docker image with the latest version of FlyreelAI's SESEMI repository. To build the image, perform the following `bash` command in this directory:

```bash
docker build \
  --build-arg USER_ID=<my_userid> \
  -t <my-sesemi>:latest .
```

You can obtain your `USER_ID` from the bash command `id -u`.
