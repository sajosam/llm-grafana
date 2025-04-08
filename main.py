import streamlit as st
import requests
import chromadb
import json
import os
from dotenv import load_dotenv


load_dotenv()

# Get API keys and Grafana key from environment variables
apikeys = os.getenv("GROQ_API_KEY", "").split(",")
grafana_key = os.getenv("GRAFANA_KEY")

# Initialize ChromaDB client
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="metrics")

# Initialize session state for metrics labels
if 'metrics_labels' not in st.session_state:
    st.session_state.metrics_labels = {}

def groqrequest(prompt):
    """Send a request to the Groq API using multiple API keys until successful or all fail."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    for key in apikeys:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key.strip()}"
        }

        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                result = response.json()
                st.write(result)
                raw_content = result["choices"][0]["message"]["content"]

                # Strip markdown code block (```json ... ```)
                if raw_content.startswith("```"):
                    raw_content = raw_content.strip("`").strip()
                    if raw_content.lower().startswith("json"):
                        raw_content = raw_content[4:].strip()

                return json.loads(raw_content)

            elif response.status_code == 401:
                st.warning(f"API key failed (unauthorized): {key.strip()}")
                continue  # Try next key
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return {"error": "API request failed"}
        except (KeyError, json.JSONDecodeError) as e:
            st.error(f"Failed to parse response: {str(e)}")
            return {"error": "Invalid API response format"}
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
            return {"error": "Request exception"}

    st.error("Daily limit reached or all API keys are invalid.")
    return {"error": "All API keys failed or rate limit reached"}


def fetch_metric_labels(metric_name):
    """Fetch unique labels for a specific metric from Prometheus"""
    url = f"http://localhost:9090/api/v1/query?query={metric_name}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            labels = set()
            for result in data.get('data', {}).get('result', []):
                labels.update(result['metric'].keys())
            labels.discard("id")
            return list(labels)
        return []
    except Exception as e:
        st.error(f"Error fetching labels for {metric_name}: {str(e)}")
        return []


# Function to fetch metrics from Prometheus
def fetch_metrics():
    url = "http://localhost:9090/api/v1/label/__name__/values"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data']
    else:
        st.error("Failed to fetch metrics from Prometheus.")
        return []


# Function to store metrics in ChromaDB
def store_metrics(metrics):
    for metric in metrics:
        collection.add(documents=[metric], ids=[metric])
    st.success(f"{len(metrics)} metrics stored successfully.")


# Function to perform a similarity search in ChromaDB
def query_metrics(user_query):
    results = collection.query(query_texts=[user_query], n_results=5)
    return results['documents'][0] if results['documents'] else []

# Function to generate PromQL query using OpenAI
def generate_promql_query(user_query, related_metrics):

    available_labels = {}
    for metric in related_metrics:
        available_labels[metric] = fetch_metric_labels(metric)
    
    prompt = f"""

        C: Context: You are querying Prometheus metrics to gather system and application performance data.

        O: Objective: Generate an accurate PromQL query based on the user's question, taking into account custom metrics if available.

        S: Style: Provide clear and concise responses tailored for Prometheus.

        T: Tone: Professional and informative.

        A: Audience: Data analysts and engineers experienced with Prometheus.

        R: Response: Format the output strictly as valid JSON (enclosed in {{}}) without any additional text or explanation.
        Here is the user's query: {user_query}

        Follow these steps to generate the query:

        1. Carefully analyze the user's question, the context, and any relevant Prometheus schema and rules.
        2. Determine if the question can be answered using general metrics knowledge (e.g., common CPU, memory, or network metrics).
        3. If the question seems complicated or involves new concepts, refer to the available custom metrics for guidance.  Use ONLY these available metrics and labels: `{json.dumps(available_labels, indent=4)}`.
        4. For straightforward or common questions, use your knowledge to generate a general PromQL query.
        5. Ensure the query is optimized for performance, compatibility with the Prometheus API, and accuracy.
        6. Strictly return at least one text field (e.g., `instance`) and an `id` field during aggregation/group by operations.
        7. Do only the task asked and avoid over-explanation or additional operations not requested.
        8. Use survey answers (if available) for filtering values in the query.
        9. Never use any labels not explicitly listed above
        10. Group by existing labels only
        11. Follow PromQL best practices
        12. Ensure the query output is formatted as a valid JSON response in the following format:
        {{
            "explanation": "A brief explanation of how you constructed the query, referencing schemas and rules as necessary.",
            "query": "The final PromQL query.format should be one of the following[query='query'&time='time' if time else '']",
            "confidence": 100
        }}

        If the question involves general metrics and you're confident, use your knowledge. If you're confused or dealing with unfamiliar metrics, use the available custom metrics as a reference.

        Here are sample user questions and their corresponding PromQL queries for reference:

        1. List all jobs:
        query: "sum by (job) (up)"
        2. List all available containers:
        query: "count(container_memory_usage_bytes) by (container_name)"
        3. Which container uses the highest CPU usage (general metric)?
        query: "topk(1, sum by (container_name) (rate(container_cpu_usage_seconds_total[5m])))"
        4. Which container is currently using the highest memory (general metric)?
        query: "topk(1, sum by (container_name) (container_memory_usage_bytes))"
        5. Which job used the highest memory in the last 1 hour (general metric)?
        query: "topk(1, avg_over_time(container_memory_usage_bytes[1h]) by (job))"

    """
    
    result = groqrequest(prompt)

    if result.get("error"):
        return {"error": "Failed to generate PromQL query from Groq API"}

    return result


def generate_grafana_dashboard(promql_response):
    """Generate Grafana dashboard JSON from PromQL response"""

    prompt = f"""
    Create Grafana 9.x dashboard JSON for this PromQL query configuration:
    {json.dumps(promql_response, indent=2)}

    Requirements:
    1. Use the given base structure for the dashboard JSON:
    {{
        "title": "Dashboard Title",
        "uid": "unique-dash-uid",
        "panels": [
            {{
                "title": "Panel Title",
                "type": "[timeseries|piechart|table]",
                "datasource": {{
                    "type": "prometheus",
                    "uid": "P1809F7CD0C75ACF3"
                }},
                "targets": [
                    {{
                        "expr": "PromQL_query",
                        "refId": "A",
                        "format": "time_series",
                        "legendFormat": "{{{{container_label_name}}}} on {{{{instance}}}}" # use this format for legend
                    }}
                ],
                "gridPos": {{
                    "h": 8,
                    "w": 12,
                    "x": 0,
                    "y": 0
                }}
            }}
        ],
        "time": {{
            "from": "now-1h",
            "to": "now"
        }},
        "refresh": "10s",
        "schemaVersion": 41,
        "version": 1
    }}

    2. Mandatory fields: uid, schemaVersion, version, gridPos for panels
    3. Use timeseries for line charts, barchart for bar charts
    4. Keep expr exactly as provided in the query
    5. Use only datasource UID: P1809F7CD0C75ACF3
    6. Set legendFormat in targets using a meaningful label (e.g. container, job, instance, name)
    7. Generate valid JSON without markdown
    8. striclty follows "legendFormat" legend_format = "{{{{container_label_name}}}} on {{{{instance}}}}"
    """

    result = groqrequest(prompt)

    if result.get("error"):
        return {"error": "Failed to generate Grafana dashboard from Groq API"}

    return result

def apply_grafana_dashboard(dashboard_json):
    """Apply dashboard to Grafana via API"""
    url = "http://localhost:3000/api/dashboards/db"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grafana_key}"
    }
    
    payload = {
        "dashboard": dashboard_json,
        "overwrite": True,
        "message": "Auto-generated by Prometheus Viewer"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            result['url'] = f"http://localhost:3000{result.get('url', '')}"
            return result
        return {"error": f"API Error {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": str(e)}


# Streamlit UI
st.title("Prometheus Metrics Viewer")

# Navigation links for APIs
st.markdown("[Go to Streamlit page](http://localhost:8501)")
st.markdown("[Go to Grafana page](http://localhost:3000)")
st.markdown("[Go to Prometheus page](http://localhost:9090)")

if st.button("Fetch Metrics"):
    metrics = fetch_metrics()
    store_metrics(metrics)

user_query = st.text_input("Enter your query (e.g., 'which container utilizes more cpus')")

if user_query:
    related_metrics = query_metrics(user_query)
    if related_metrics:
        st.write("Related Metrics:")
        for metric in related_metrics:
            st.write(metric)
        
        promql_response = generate_promql_query(user_query, related_metrics)
        
        # Store the response in session state for later use
        st.session_state.promql_response = promql_response
        st.json(promql_response)

        # Dashboard creation section - use a form to prevent rerun
        with st.form(key='dashboard_form'):
            if st.form_submit_button("Create Grafana Dashboard"):
                if 'promql_response' in st.session_state:
                    dashboard_json = generate_grafana_dashboard(st.session_state.promql_response)
                    
                    if dashboard_json:
                        st.subheader("Generated Dashboard")
                        st.json(dashboard_json)
                        st.session_state.dashboard_json = dashboard_json
                    else:
                        st.error("Failed to generate dashboard JSON")
            
            # Apply to Grafana
            if 'dashboard_json' in st.session_state:
                response = apply_grafana_dashboard(st.session_state.dashboard_json)
                if response.get("error"):
                    st.error(f"Failed to apply dashboard: {response.get('error')}")
                else:
                    st.success(f"Dashboard applied successfully. Access it at {response['url']}")

