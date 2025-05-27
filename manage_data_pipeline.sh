#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
if [ -f deploy.env ]; then
    echo "Sourcing environment variables from deploy.env"
    set -a # automatically export all variables
    source deploy.env
    set +a 
fi

# Ensure critical variables are set, otherwise exit
AZURE_RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:?Error: AZURE_RESOURCE_GROUP is not set. Please set it in deploy.env or as an environment variable.}"
COSMOS_DB_ACCOUNT_NAME="${COSMOS_DB_ACCOUNT_NAME:?Error: COSMOS_DB_ACCOUNT_NAME is not set.}"
COSMOS_DB_DATABASE_NAME="${COSMOS_DB_DATABASE_NAME:?Error: COSMOS_DB_DATABASE_NAME is not set.}"
WEB_APP_NAME="${AZURE_APP_NAME:?Error: AZURE_APP_NAME is not set.}"

# Get Web App URL
echo ""
echo "🔎 Retrieving Web App URL for '$WEB_APP_NAME'..."
WEB_APP_URL_RAW=$(az webapp show --resource-group "$AZURE_RESOURCE_GROUP" --name "$WEB_APP_NAME" --query "defaultHostName" -o tsv 2>/dev/null)

if [ -z "$WEB_APP_URL_RAW" ]; then
    echo "❌ Error: Could not retrieve Web App URL for '$WEB_APP_NAME'."
    echo "   Please ensure the Web App is deployed and \`az login\` is successful." >&2
    exit 1
fi
WEB_APP_URL="https://$WEB_APP_URL_RAW"
echo "   ✅ Web App URL: $WEB_APP_URL"


# Collections to ensure exist (collection_name: shard_key_path_for_mongodb)
declare -A COLLECTIONS_TO_ENSURE=(
    ["users"]="username"
    ["api_keys"]="user_id"
    ["astronomical_objects"]="catalog_source"
    ["astronomical_data"]="object_id"
    ["search_history"]="user_id"
    ["ml_analysis_results"]="user_id"
    ["api_cache"]="_id"
    ["system_statistics"]="metric_name"
)

# --- Functions ---

check_azure_login() {
    echo ""
    echo "🔗 Checking Azure login status..."
    if ! az account show &> /dev/null; then
        echo "   Attempting to login to Azure using device code..."
        az login --use-device-code
    fi
    echo "   ✅ Azure login verified."
}

ensure_database_exists() {
    echo ""
    echo "Ensuring database '$COSMOS_DB_DATABASE_NAME' in account '$COSMOS_DB_ACCOUNT_NAME' exists..."
    if az cosmosdb mongodb database show --account-name "$COSMOS_DB_ACCOUNT_NAME" --name "$COSMOS_DB_DATABASE_NAME" --resource-group "$AZURE_RESOURCE_GROUP" &> /dev/null; then
        echo "   ✅ Database '$COSMOS_DB_DATABASE_NAME' already exists."
    else
        echo "   Database '$COSMOS_DB_DATABASE_NAME' not found. Creating..."
        az cosmosdb mongodb database create --account-name "$COSMOS_DB_ACCOUNT_NAME" --name "$COSMOS_DB_DATABASE_NAME" --resource-group "$AZURE_RESOURCE_GROUP"
        echo "   ✅ Database '$COSMOS_DB_DATABASE_NAME' created."
    fi
}

ensure_collections_exist() {
    echo ""
    echo "Ensuring collections exist and shard keys are configured (if creating new)..."
    for collection_name in "${!COLLECTIONS_TO_ENSURE[@]}"; do
        shard_key_path="${COLLECTIONS_TO_ENSURE[$collection_name]}"
        echo "  Checking collection: '$collection_name' with intended shard key: '$shard_key_path'..."
        
        if az cosmosdb mongodb collection show --account-name "$COSMOS_DB_ACCOUNT_NAME" --database-name "$COSMOS_DB_DATABASE_NAME" --name "$collection_name" --resource-group "$AZURE_RESOURCE_GROUP" &> /dev/null; then
            echo "    ✅ Collection '$collection_name' already exists."
        else
            echo "    Collection '$collection_name' not found. Creating with shard key '$shard_key_path'..."
            az cosmosdb mongodb collection create \
                --account-name "$COSMOS_DB_ACCOUNT_NAME" \
                --database-name "$COSMOS_DB_DATABASE_NAME" \
                --name "$collection_name" \
                --resource-group "$AZURE_RESOURCE_GROUP" \
                --shard "$shard_key_path" \
            echo "    ✅ Collection '$collection_name' created."
        fi
    done
}

trigger_api_endpoint() {
    local endpoint_path="$1"
    local description="$2"
    local method="${3:-POST}"
    local payload_json="${4:-{\}}"

    echo ""
    echo "🚀 TRIGGERING: $description ($method $WEB_APP_URL$endpoint_path)"
    
    TMP_RESPONSE_FILE=$(mktemp)
    
    http_status=$(curl -s -L -X "$method" \
        -H "Content-Type: application/json" \
        -d "$payload_json" \
        --connect-timeout 15 --max-time 60 \
        -o "$TMP_RESPONSE_FILE" \
        -w "%{http_code}" \
        "$WEB_APP_URL$endpoint_path")
    
    body=$(cat "$TMP_RESPONSE_FILE")
    rm -f "$TMP_RESPONSE_FILE"

    echo "   Response Body: $body"
    echo "   HTTP Status: $http_status"

    if [[ "$http_status" -ge 200 && "$http_status" -lt 300 ]]; then
        echo "   ✅ $description successfully triggered."
        task_id=$(echo "$body" | jq -r '.task_id // .job_id // "unknown_task_id"')
        echo "   Task ID: $task_id"
        if [ "$task_id" != "unknown_task_id" ] && [ "$task_id" != "null" ] && [ -n "$task_id" ]; then
            monitor_task "$task_id"
        else 
            echo "   ⚠️ No valid Task ID found in response to monitor."
        fi
    else
        echo "   ❌ Error triggering $description. Status: $http_status. Body: $body"
    fi
}

monitor_task() {
    local task_id="$1"
    echo "   🔄 Monitoring task '$task_id'... (Press Ctrl+C to stop monitoring early)"
    
    local status="started"
    local progress=0
    local attempts=0
    local max_attempts=${MAX_MONITOR_ATTEMPTS:-60} 
    local sleep_interval=${MONITOR_SLEEP_INTERVAL:-10}

    while [[ "$status" != "completed" && "$status" != "failed" && "$attempts" -lt "$max_attempts" ]]; do
        sleep "$sleep_interval"
        attempts=$((attempts+1))
        
        task_info_response=$(curl -s -L -X GET --connect-timeout 10 --max-time 20 "$WEB_APP_URL/api/heavy/task-status/$task_id" -w "HTTP_STATUS_CODE:%{http_code}")
        task_http_status=$(echo "$task_info_response" | sed -n 's/.*HTTP_STATUS_CODE://p')
        task_info_body=$(echo "$task_info_response" | sed 's/HTTP_STATUS_CODE:.*//')

        if [[ "$task_http_status" -ne 200 ]]; then
            echo "    [Attempt $attempts/$max_attempts] Error fetching status for task '$task_id'. HTTP Status: $task_http_status. Response: $task_info_body"
            if [[ "$attempts" -gt 5 && "$task_http_status" -ne 404 ]]; then
                 echo "    ⚠️  Persistent error fetching task status. Stopping monitor."
                 break
            fi
            continue
        fi
        
        status=$(echo "$task_info_body" | jq -r '.status // "unknown"')
        progress=$(echo "$task_info_body" | jq -r '.progress // 0')
        message=$(echo "$task_info_body" | jq -r '.message // "-"')
        
        echo "    [Attempt $attempts/$max_attempts] Task '$task_id' status: $status ($progress%) - $message"
        
        if [[ "$status" == "completed" ]]; then
            echo "   ✅ Task '$task_id' completed successfully."
            echo "   Details: $task_info_body"
            break
        elif [[ "$status" == "failed" ]]; then
            echo "   ❌ Task '$task_id' failed."
            echo "   Details: $task_info_body"
            break
        fi
    done

    if [[ "$attempts" -ge "$max_attempts" && "$status" != "completed" && "$status" != "failed" ]]; then
        echo "   ⚠️  Monitoring for task '$task_id' timed out after $max_attempts attempts. Check status manually via API or Azure logs."
    fi
}

# --- Main Script Logic ---
check_azure_login
echo ""
echo "--- Step 1: Verifying/Creating Database and Collections ---" 
ensure_database_exists
ensure_collections_exist

echo ""
echo "--- Step 2: Triggering Catalog Data Preprocessing Pipeline ---"
trigger_api_endpoint "/api/heavy/astro/trigger-preprocessing" "Full catalog preprocessing"

Example for custom data processing (Uncomment to activate)
echo ""
echo "--- Step 3: Triggering Custom Data Processing (Example: clean) ---"
CUSTOM_CONFIG_JSON='{"processing_type": "clean"}'
trigger_api_endpoint "/api/heavy/data/custom-process" "Custom data cleaning" "POST" "$CUSTOM_CONFIG_JSON"

echo ""
echo "🎉 Pipeline management script finished."