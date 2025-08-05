# Log ML Service


## 🔍 File Descriptions

### `demo.py`

* Implements the **FastAPI** server that loads a pre-trained ML model (`model.joblib`) for inference on the Iris dataset.
* **Endpoints**:

  * `POST /predict`: Accepts Iris flower measurements and returns a prediction with confidence.
  * `GET /live_check`: Liveness probe to confirm the app is running.
  * `GET /ready_check`: Readiness probe to confirm the model is loaded and ready to serve.
* **Telemetry & Logging**:

  * Uses **OpenTelemetry** and **Google Cloud Trace** for distributed tracing.
  * Logs are structured and JSON-formatted for compatibility with logging systems.
  * Request latency is captured and added to the response headers (`X-Process-Time-ms`).
* **Error Handling**:

  * Captures and logs unhandled exceptions with trace IDs.

---

### `Dockerfile`

* Containerizes the FastAPI app.
* Installs dependencies from `requirements.txt`.
* Exposes port `8200` and runs the app with `uvicorn`.

---

### `deployment.yaml`

* Defines a Kubernetes **Deployment** with:

  * 2 replicas of the app.
  * Health probes (`/live_check`, `/ready_check`) for lifecycle management.
  * Resource requests and limits to optimize scheduling.
  * Uses a custom image hosted in Google Artifact Registry.

---

### `service.yaml`

* Exposes the deployment as a Kubernetes **LoadBalancer Service**.
* Maps external port `80` to container port `8200`.

---

### `hpa.yaml`

* Configures a **Horizontal Pod Autoscaler (HPA)**:

  * Auto-scales the pods between 2 and 10 replicas.
  * Targets CPU utilization at 60% for scaling decisions.

---
