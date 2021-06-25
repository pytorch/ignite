import json
import os
import sys
import time

import requests


def assert_result(result, expected_code):
    if result.status_code != expected_code:
        raise RuntimeError(f"{result.url}, {result.status_code}: {result.text}")


def get_output(result_text, required_keys):
    output = json.loads(result_text)

    if not all([v in output for v in required_keys]):
        raise RuntimeError(f"Output does not contain required fields: {required_keys}\n" f"Output is: {output}")
    return output


def trigger_new_pipeline(data, headers):
    result = requests.post(
        "https://circleci.com/api/v2/project/gh/pytorch/ignite/pipeline", data=json.dumps(data), headers=headers
    )
    assert_result(result, 201)
    output = get_output(result.text, ["id",])
    return output["id"]


def assert_pipeline_created(pipeline_id, headers):
    while True:
        result = requests.get(f"https://circleci.com/api/v2/pipeline/{pipeline_id}", headers=headers)
        assert_result(result, 200)
        output = get_output(result.text, ["state", "errors"])

        if output["state"] == "errored":
            raise RuntimeError(f"Pipeline is errored: {output['errors']}")
        if output["state"] == "created":
            break
        time.sleep(2)


def get_workflow_id(pipeline_id, headers):

    while True:
        result = requests.get(f"https://circleci.com/api/v2/pipeline/{pipeline_id}/workflow", headers=headers)
        assert_result(result, 200)
        output = get_output(result.text, ["items",])
        items = output["items"]
        if len(items) > 1:
            raise RuntimeError(f"Incorrect number of workflow ids: {len(items)} != 1\n" f"items: {items}")
        if len(items) < 1:
            continue
        item_0 = items[0]
        if "id" not in item_0:
            raise RuntimeError("Workflow info does not contain 'id'\n" f"Info: {item_0}")
        return item_0["id"]


def assert_workflows_successful(pipeline_id, headers):

    workflow_id = get_workflow_id(pipeline_id, headers)

    base_url = "https://app.circleci.com/pipelines/github/pytorch/ignite"
    url = None

    while True:
        result = requests.get(f"https://circleci.com/api/v2/workflow/{workflow_id}", headers=headers)
        assert_result(result, 200)
        output = get_output(result.text, ["name", "status", "pipeline_number"])

        if url is None:
            url = f"{base_url}/{output['pipeline_number']}/workflows/{workflow_id}"
            print(f"Circle CI workflow: {url}")

        if output["status"] in ["error", "failing", "canceled", "not_run", "failed"]:
            raise RuntimeError(f"Workflow failed: {output['status']}\n" f"See {url}")
        if output["status"] == "success":
            print("\nWorkflow successful")
            break
        time.sleep(30)
        print(".", end=" ")


if __name__ == "__main__":

    print("Trigger new pipeline on Circle-CI")

    if "CIRCLE_TOKEN" not in os.environ:
        raise RuntimeError(
            "Can not find CIRCLE_TOKEN env variable.\nPlease, export CIRCLE_TOKEN=<token> before calling this script."
            "This token should be a user token and not the project token."
        )
        # https://discuss.circleci.com/t/triggering-pipeline-via-v2-api-fails-with-404-project-not-found/39342/2

    argv = sys.argv
    if len(argv) != 3:
        raise RuntimeError("Usage: python trigger_circle_ci.py <true or false> <branch-name>")

    should_publish_docker_images = json.loads(argv[1])
    branch = argv[2]

    print(f"- should_publish_docker_images: {should_publish_docker_images}")
    print(f"- Branch: {branch}")
    if branch.startswith("refs/pull") and branch.endswith("/merge"):
        branch = branch.replace("/merge", "/head")
        print(f"Replaced /merge -> /head : {branch}")

    headers = {"authorization": "Basic", "content-type": "application/json", "Circle-Token": os.environ["CIRCLE_TOKEN"]}

    data = {
        "branch": branch,
        "parameters": {
            "should_build_docker_images": True,
            "should_publish_docker_images": should_publish_docker_images,
        },
    }

    unique_pipeline_id = trigger_new_pipeline(data, headers)
    assert_pipeline_created(unique_pipeline_id, headers)
    assert_workflows_successful(unique_pipeline_id, headers)
