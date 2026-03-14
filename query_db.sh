#!/bin/bash
# Usage: ./query_db.sh "SELECT * FROM session_progress"
# Direct RDS connection - instant, no ECS task needed

SQL="${1:-SELECT access_token, current_phase, search_completed, flight_selection_confirmed, updated_at FROM session_progress ORDER BY updated_at DESC LIMIT 20}"

python3 - "$SQL" <<'EOF'
import sys, os
import pymysql

sql = sys.argv[1]

conn = pymysql.connect(
    host='flight-ranker-db.c274giyammku.us-east-1.rds.amazonaws.com',
    user='appuser',
    password='AppUserRDS2024',
    database='flight_rankings',
    port=3306,
    ssl={'ssl_ca': None},
    ssl_verify_cert=False
)

try:
    cur = conn.cursor()
    cur.execute(sql)
    if cur.description:
        print('\t'.join(d[0] for d in cur.description))
        for row in cur.fetchall():
            print('\t'.join(str(x) if x is not None else 'NULL' for x in row))
    print(f'--- {cur.rowcount} rows ---')
finally:
    conn.close()
EOF
