
import os
import json

assert "CLUSTER_SPEC" in os.environ

cluster_spec = json.loads(os.environ['CLUSTER_SPEC'])

master_addr_port = cluster_spec['cluster']['master'][0].split(":")
master_addr = master_addr_port[0]
master_port = master_addr_port[1]

rank = cluster_spec['task']['index']
if cluster_spec['task']['type'] == "worker":
    rank += 1

print("{},{},{}".format(master_addr, master_port, rank))
