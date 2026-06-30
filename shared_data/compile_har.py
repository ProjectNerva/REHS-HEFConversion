import sys
from hailo_sdk_client import ClientRunner

har_name = sys.argv[1]
hef_name = sys.argv[2]

runner = ClientRunner(har=f'{har_name}.har')
hef = runner.compile()
with open(f"{hef_name}.hef", "wb") as f:
    f.write(hef)
