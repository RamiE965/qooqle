"""
Utility script to inspect the status and results of a remote Qiskit Runtime job.

This script:
  1. Configures the QiskitRuntimeService account (for the IBM Quantum Platform).
  2. Connects to IBM Quantum.
  3. Retrieves a specific job by ID.
  4. Prints the job status.
  5. If completed, pulls the result and extracts the best bitstring from a
     QAOA-style eigenstate result.

Intended use:
  - Debugging / post-mortem inspection of a specific job.
  - Quickly re-running result extraction logic without re-submitting the job.

Note:
  - Avoid hard-coding API tokens in source files. Prefer environment variables
    or a secure secrets manager, especially for shared repos.
"""

from qiskit_ibm_runtime import QiskitRuntimeService
import os

# The Job ID from your previous run (e.g., copied from logs or portal UI).
JOB_ID = "d4bm8bhlag1s73blcvp0"

# ⚠️ SECURITY WARNING:
# Do not hard-code real API tokens in committed source code.
# Instead, read from an environment variable or a secure secret store.
# Example: export IBM_QUANTUM_TOKEN="your-real-token"
api_token = os.getenv("IBM_QUANTUM_TOKEN", "<YOUR_IBM_QUANTUM_TOKEN_HERE>")

# Persist the IBM Quantum Platform account configuration locally.
# - channel: selects which IBM Quantum entry point to use.
# - token:   the authentication token associated with your IBMid account.
# - overwrite=True: update any existing stored account with these credentials.
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=api_token,
    overwrite=True,
)

print(f"Connecting to IBM Quantum to retrieve job: {JOB_ID}...")

# 1. Initialize the runtime service for the specific IBM Quantum instance.
#    The `instance` string encodes the cloud account / resource group / project.
service = QiskitRuntimeService(
    instance=(
        "crn:v1:bluemix:public:quantum-computing:us-east:"
        "a/962b41b03d3a4cff935c780b7cd80779:"
        "7fe9e952-f9e8-41b7-8c7f-8217f87e6661::"
    )
)

# 2. Retrieve the job handle from the remote backend using the Job ID.
job = service.job(JOB_ID)

# Print a human-readable status (e.g., QUEUED, RUNNING, DONE, ERROR).
status = job.status()
print(f"Job status: {status}")

if status == "DONE":
    # 3. Pull the result payload from the completed job.
    #    The structure depends on the program that was run; here we assume it
    #    returns eigenstates compatible with QAOA-style post-processing.
    result = job.result()
    print("\n--- JOB RESULT ---")
    print(result)
    
    # Attempt to extract the "best" bitstring from the eigenstate distribution.
    # We assume:
    #   - result[0] is the first (and relevant) result object.
    #   - result[0].data.eigenstate.binary_probabilities() returns a dict-like
    #     mapping {bitstring: probability}.
    try:
        eigen_probs = result[0].data.eigenstate.binary_probabilities()
        # Select the bitstring with the highest probability.
        best_bitstring_str = max(eigen_probs, key=eigen_probs.get)

        print("\n--- BEST RESULT ---")
        print(f"Best Bitstring: {best_bitstring_str}")
    except Exception as e:
        # If the result format doesn't match expectations (e.g., different
        # program output schema), we surface a helpful error rather than crash.
        print(f"Could not parse eigenstate from result payload: {e}")
else:
    # For jobs that are still queued/running or ended in an error state.
    print("Job is not yet complete or did not finish successfully.")
    print("Check again later or inspect the job details in the IBM Quantum UI.")
