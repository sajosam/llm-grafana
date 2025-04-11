import streamlit as st
import requests
import chromadb
import json
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
import docker


load_dotenv()

# Get API keys and Grafana key from environment variables
apikeys = os.getenv("GROQ_API_KEY", "").split(",")
grafana_key = os.getenv("GRAFANA_KEY")
# prometheus_host = os.getenv("PROMETHEUS_HOST", "http://localhost")
prometheus_host = os.getenv("PROMETHEUS_HOST", "http://localhost")


# Initialize ChromaDB client
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="metrics")

# Initialize session state for metrics labels
if 'metrics_labels' not in st.session_state:
    st.session_state.metrics_labels = {}

def get_exposed_port(container_name):
    client = docker.from_env()
    try:
        # Get the container by its name
        container = client.containers.get(container_name)
    except docker.errors.NotFound:
        print(f"Container '{container_name}' not found.")
        return None

    # Retrieve port mapping details from the container's attributes
    ports = container.attrs['NetworkSettings']['Ports']
    for container_port, host_bindings in ports.items():
        if host_bindings:  # Check if port is exposed
            # Return the first host port found in the bindings
            return host_bindings[0]['HostPort']

    # If no exposed ports found, return None
    return None

def adjust_prometheus_url(original_url):
    """
    Adjust the Prometheus URL based on environment configuration or via Docker inspection.
    
    Steps:
      1. Parses the original URL (e.g., "https://prometheus:9090") and extracts the hostname
         to use as the container name.
      2. Checks for an environment variable `PROMETHEUS_HOST` to override URL details.
         If defined, it extracts the host (and possibly a port) from it.
      3. If no environment variable is set, it calls get_exposed_port using the container 
         name parsed from the original URL to get the actual exposed host port.
      4. Rebuilds the URL with the new host and port as determined.
    """
    original_parsed = urlparse(original_url)
    # Parse container name from the original URL's hostname
    container_name = original_parsed.hostname

    # Check for an override using PROMETHEUS_HOST environment variable
    prometheus_host = os.getenv("PROMETHEUS_HOST", "http://localhost").strip()
    if prometheus_host:
        # Determine if PROMETHEUS_HOST includes a scheme
        if prometheus_host.startswith(('http://', 'https://')):
            parsed_env = urlparse(prometheus_host)
            new_host = parsed_env.hostname
            new_port = get_exposed_port(container_name)
        else:
            if ':' in prometheus_host:
                new_host, port_part = prometheus_host.split(':', 1)
                new_port = int(port_part) if port_part.isdigit() else None
            else:
                new_host = prometheus_host
                new_port = None
        # Use new_port if provided, otherwise fall back to the port in the original URL.
        final_port = new_port or original_parsed.port
    else:
        # If no environment override exists, use Docker SDK to inspect the container's port mapping.
        exposed_port = get_exposed_port(container_name)
        new_host = container_name  # Keep the container name as the host
        final_port = exposed_port if exposed_port else original_parsed.port

    # Rebuild the netloc (host:port) portion.
    if final_port:
        new_netloc = f"{new_host}:{final_port}"
    else:
        new_netloc = new_host

    return original_parsed._replace(netloc=new_netloc).geturl()

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


def fetch_metrics(prom_url):
    """Fetch metrics from specific Prometheus instance"""
    try:
        response = requests.get(f"{prom_url}/api/v1/label/__name__/values", timeout=10)
        return response.json().get('data', []) if response.ok else []
    except Exception as e:
        st.error(f"Metrics fetch failed: {str(e)}")
        return []


def store_metrics(metrics, ds_uid):
    """Store metrics in datasource-specific collection"""
    collection = db.get_or_create_collection(name=ds_uid)
    existing = collection.get()['ids']
    new_metrics = [m for m in metrics if m not in existing]
    
    if new_metrics:
        collection.add(documents=new_metrics, ids=new_metrics)
    return len(new_metrics)


def query_metrics(user_query, ds_uid):
    """Query datasource-specific collection"""
    try:
        collection = db.get_collection(name=ds_uid)
        results = collection.query(query_texts=[user_query], n_results=5)
        return results['documents'][0]
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return []

# Function to generate PromQL query using OpenAI
def generate_promql_query(user_query_map):

    
    prompt = f"""
        Context:You are generating PromQL queries to retrieve system and application metrics from Prometheus.

        Objective:Create accurate, optimized PromQL queries strictly using the provided input. Prioritize custom metrics and apply only the given labels.

        Style:lear, minimal, and Prometheus-friendly.

        Tone:Professional and concise.

        Audience:Engineers and analysts experienced with Prometheus.

        Response:Return a valid JSON object only — no extra text or explanation.

        Input array:  
        {json.dumps(user_query_map, indent=4)}

        Guidelines:

        1. For each item:
        - Use the `mandatory_datasource_uuid` (required).
        - Use only the `mandatory_similar_metrics`. **Do not use any other metrics.**
        - Use only the `mandatry_corresponding_metrics_labels`. **Do not add or infer labels.**

        2. Follow PromQL best practices:
        - Use text-based identifiers (e.g., `instance`, `job`) and include an `id` label when aggregating.
        - Group only by provided labels.
        - Ensure performance and correctness.

        3. Format output as:
        {{
            "result": [
                {{
                    "mandatory_datasource_uuid": "value",
                    "userquery": "value",
                    "query": "Generated PromQL query"
                }}
            ]
        }}

        Example queries:

        - List all containers:  
        `"count(container_memory_usage_bytes) by (container_name)"`

        - Highest CPU container:  
        `"topk(1, sum by (container_name) (rate(container_cpu_usage_seconds_total[5m])))"`

        - Redis clients:  
        `"redis_connected_clients"`

        - Node CPU:  
        `"node_cpu_seconds_total"`
    """

    result = groqrequest(prompt)

    if result.get("error"):
        return {"error": "Failed to generate PromQL query from Groq API"}

    return result


def generate_grafana_dashboard(promql_response):
    """
    Generate a Grafana dashboard JSON from the PromQL responses.
    The promql_response is expected to have the following structure:
    {
        "result": [
            {
                "datasource_uuid": "...",
                "userquery": "...",
                "query": "Generated PromQL query"
            },
            ...
        ]
    }
    The dashboard JSON will have one panel per query.
    """
    prompt = f"""
    Create Grafana 9.x dashboard JSON for the following PromQL query configuration:
    {json.dumps(promql_response, indent=2)}

    Requirements:
    1. Use the following base structure for the dashboard JSON:
    {{
        "title": "Dashboard Title",
        "uid": "unique-dash-uid",
        "panels": [ 
            // Create one panel for each entry below 
        ],
        "time": {{
            "from": "now-1h",
            "to": "now"
        }},
        "refresh": "10s",
        "schemaVersion": 41,
        "version": 1
    }}

    2. For each object in promql_response["result"], create a panel with:
       - "title": Use the "userquery" as inspiration for the panel title.
       - "type": [timeseries|piechart|table|linechart|guage].
       - "datasource": {{
             "type": "prometheus",
             "uid": the value from "datasource_uuid" for that object
         }}
       - "targets": An array containing one object with:
             "expr": the PromQL query from the "query" field,
             "refId": "A",
             "format": "[timeseries|piechart|table|linechart|guage]",
             "legendFormat": "{{{{container_label_name}}}} on {{{{instance}}}}"
       - "gridPos": Allocate grid positions sequentially for each panel.
         For example: The first panel at {{"x": 0, "y": 0, "w": 12, "h": 8}},
         the second at {{"x": 12, "y": 0, "w": 12, "h": 8}}, the third at {{"x": 0, "y": 8, "w": 12, "h": 8}}, etc.
    
    3. The output must be a valid JSON object without markdown formatting or additional text.
    4. Mandatory fields: uid, schemaVersion, version, gridPos for panels
    5. Keep expr exactly as provided in the query
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

def fetch_datasources():
    """Fetch and process Prometheus datasources from Grafana"""
    url = "http://localhost:3000/api/datasources"
    headers = {"Authorization": f"Bearer {grafana_key}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            processed_ds = []
            for ds in response.json():
                # Only process Prometheus datasources
                if ds.get('type') == 'prometheus':
                    # Create a copy to avoid modifying original data
                    modified_ds = ds.copy()
                    modified_ds['adjusted_url'] = adjust_prometheus_url(ds.get('url', ''))
                    processed_ds.append(modified_ds)
            return processed_ds
        st.error("Failed to fetch datasources")
        return []
    except Exception as e:
        st.error(f"Datasource error: {str(e)}")
        return []


# Streamlit UI
st.title("Prometheus Metrics Dashboard Builder")

datasources = fetch_datasources()
if not datasources:
    st.warning("No Prometheus datasources found in Grafana")
    st.stop()

ds_options = {ds['name']: (ds['uid'], ds['adjusted_url']) for ds in datasources}

with st.expander("Metric Management", expanded=False):
    if st.button("Refresh All Metrics"):
        with st.spinner("Updating metrics..."):
            for ds in datasources:
                metrics = fetch_metrics(ds['adjusted_url'])
                if metrics:
                    count = store_metrics(metrics, ds['uid'])
                    st.success(f"Updated {ds['name']} with {count} new metrics")

st.header("Create New Dashboard")
queries = []

with st.form("dashboard_form"):
    for i in range(3):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(f"Query {i+1}", key=f"q{i}")
        with col2:
            ds_name = st.selectbox(f"Datasource {i+1}", options=ds_options.keys(), key=f"ds{i}")
        queries.append((query, ds_name))


    if st.form_submit_button("Generate Dashboard"):
        processed_queries = []
        for query, ds_name in queries:
            if query and ds_name:
                ds_uid, ds_url = ds_options[ds_name]
                similar = query_metrics(query, ds_uid)
                labels = {}
                
                for metric in similar:
                    label_res = requests.get(f"{ds_url}/api/v1/query?query={metric}")
                    if label_res.ok:
                        # Get the full response data
                        response_data = label_res.json().get('data', {})
                        results = response_data.get('result', [])

                        # Check if there are any results before accessing
                        if results:
                            # Convert keys to a set, discard "id", then convert back to list
                            keys = set(results[0].get('metric', {}).keys())
                            keys.discard("id")
                            labels[metric] = list(keys)
                        else:
                            labels[metric] = []
                            st.warning(f"No results found for metric: {metric}")
                
                processed_queries.append({
                    "mandatory_datasource_uuid": ds_uid,
                    "userquery": query,
                    "mandatory_similar_metrics": similar,
                    "mandatry_corresponding_metrics_labels": labels
                })

        if processed_queries:
            with st.spinner("Generating queries..."):
                promql_response = generate_promql_query(processed_queries)
                
            if not promql_response.get('error'):
                with st.spinner("Creating dashboard..."):
                    dashboard_json = generate_grafana_dashboard(promql_response)
                    
                if not dashboard_json.get('error'):
                    apply_response = apply_grafana_dashboard(dashboard_json)
                    if 'url' in apply_response:
                        st.success(f"Dashboard created: [View Dashboard]({apply_response['url']})")
                    else:
                        st.error("Failed to deploy dashboard")
                else:
                    st.error("Dashboard generation failed")
            else:
                st.error("Query generation failed")