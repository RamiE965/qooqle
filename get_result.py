from qiskit_ibm_runtime import QiskitRuntimeService

# The Job ID from your log and screenshot
JOB_ID = "d4bm8bhlag1s73blcvp0"
api_token = "62oyaPIhvWNAUhVSnMV4T928Q_g8PTmEYoA8rCmdO_gV"
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=api_token, overwrite=True)

print(f"Connecting to IBM Quantum to retrieve job: {JOB_ID}...")

# 1. Get a FRESH authentication token
service = QiskitRuntimeService(
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/962b41b03d3a4cff935c780b7cd80779:7fe9e952-f9e8-41b7-8c7f-8217f87e6661::"
)

# 2. Retrieve the job from the cloud
job = service.job(JOB_ID)

print(f"Job status: {job.status()}")

if job.status() == "DONE":
    # 3. Get the results
    result = job.result()
    print("\n--- JOB RESULT ---")
    print(result)
    
    # This part is from your QAOA logic to find the best bitstring
    try:
        best_bitstring_str = max(
            result[0].data.eigenstate.binary_probabilities(),
            key=result[0].data.eigenstate.binary_probabilities().get
        )
        print("\n--- BEST RESULT ---")
        print(f"Best Bitstring: {best_bitstring_str}")
    except Exception as e:
        print(f"Could not parse eigenstate: {e}")
else:
    print("Job is not yet complete. Try again in a few minutes.")