#!/bin/bash
# Usage: ./query_db.sh "SELECT * FROM session_progress"
export AWS_PROFILE=personal

SQL="${1:-SELECT access_token, current_phase, search_completed, flight_selection_confirmed, updated_at FROM session_progress ORDER BY updated_at DESC LIMIT 20}"

# Write Python script to temp file and base64-encode it to avoid JSON quoting issues
PYTHON_SCRIPT=$(cat <<'PYEOF'
import sys, os
sys.path.insert(0, '/app')
os.chdir('/app')
from backend.db import SessionLocal
import sqlalchemy
sql = os.environ['QUERY_SQL']
db = SessionLocal()
try:
    rows = db.execute(sqlalchemy.text(sql)).fetchall()
    for r in rows:
        print('|'.join(str(c) for c in r))
    print(f'--- {len(rows)} rows ---')
finally:
    db.close()
PYEOF
)

# Base64-encode the script (no newlines) to safely embed in JSON
B64=$(echo "$PYTHON_SCRIPT" | base64 | tr -d '\n')

# Escape the SQL for JSON (replace backslash, double-quote, newline, tab)
SQL_JSON=$(echo "$SQL" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))")

SUBNET=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
  --query 'Subnets[0].SubnetId' --output text --region us-east-1)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=flight-ranker-sg \
  --query 'SecurityGroups[0].GroupId' --output text --region us-east-1)

# Build overrides JSON using python3 to avoid any shell quoting issues
OVERRIDES=$(python3 -c "
import json
b64 = '$B64'
sql = $SQL_JSON
overrides = {
    'containerOverrides': [{
        'name': 'flight-ranker',
        'command': ['python3', '-c', f'import base64,os; exec(base64.b64decode(\"{b64}\").decode())'],
        'environment': [{'name': 'QUERY_SQL', 'value': sql}]
    }]
}
print(json.dumps(overrides))
")

TASK=$(aws ecs run-task \
  --cluster flight-ranker-cluster \
  --task-definition flight-ranker-task \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET],securityGroups=[$SG],assignPublicIp=ENABLED}" \
  --overrides "$OVERRIDES" \
  --region us-east-1 \
  --query 'tasks[0].taskArn' --output text)

if [ -z "$TASK" ] || [ "$TASK" = "None" ]; then
  echo "ERROR: Failed to start ECS task"
  exit 1
fi

TASK_ID=${TASK##*/}
echo "Running query... (task $TASK_ID)"

while true; do
  STATUS=$(aws ecs describe-tasks --cluster flight-ranker-cluster --tasks $TASK_ID \
    --region us-east-1 --query 'tasks[0].lastStatus' --output text 2>/dev/null)
  [ "$STATUS" = "STOPPED" ] && break
  sleep 5
done

aws logs filter-log-events \
  --log-group-name /ecs/flight-ranker \
  --log-stream-names "ecs/flight-ranker/$TASK_ID" \
  --region us-east-1 \
  --query 'events[*].message' --output text \
  | grep -v "Warning\|enableCORS\|enableXsrf\|cross-origin\|cookie\|More information\|order to protect\|disable server"
