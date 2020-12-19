#!/bin/bash

echo "Trigger new pipeline on Circle-CI"

if [ -z ${CIRCLE_TOKEN} ]; then
    echo "Can not find CIRCLE_TOKEN env variable"
    echo "Please, export CIRCLE_TOKEN=<token> before calling this script"
    exit 1
fi


if [ -z "$1" ]; then
  echo "Boolean should_publish_docker_images should be provided. Usage: sh trigger_circle_ci.sh <true or false> <branch-name>"
  exit 1
fi

should_publish_docker_images=$1

if [ -z "$2" ]; then
  exit 1
fi

branch=$2

echo "- should_publish_docker_images: ${should_publish_docker_images}"
echo "- Branch: ${branch}"

curl --request POST \
  --url https://circleci.com/api/v2/project/gh/pytorch/ignite/pipeline \
  --header "authorization: Basic" \
  --header "content-type: application/json" \
  --header "Circle-Token: ${CIRCLE_TOKEN}" \
  --data "{\"branch\":\"$branch\",\"parameters\":{\"should_build_docker_images\":true,\"should_publish_docker_images\":$should_publish_docker_images}}"
