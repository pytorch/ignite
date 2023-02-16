---
title: Scheduled workflow failed
labels:
  - bug
---

Oh no, something went wrong in the scheduled workflow **{{ env.GITHUB_WORKFLOW }} with commit {{ env.GITHUB_SHA }}**.
Please look into it:

{{ env.GITHUB_SERVER_URL }}/{{ env.GITHUB_REPOSITORY }}/actions/runs/{{ env.GITHUB_RUN_ID }}

Feel free to close this if this was just a one-off error.
