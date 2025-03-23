#!/bin/bash
docker rm -f ros2_playground 2>/dev/null
docker run -it --name ros2_playground \
  -v /mnt/ssd/BFV_project:/root/code \
  ros:humble