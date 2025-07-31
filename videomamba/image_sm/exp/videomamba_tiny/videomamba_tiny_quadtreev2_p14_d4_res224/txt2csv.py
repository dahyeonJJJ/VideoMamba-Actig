#!/usr/bin/env python3
import json
import csv

INPUT  = "log.txt"          # 원본 로그 파일
OUTPUT = "log_out.csv" # 만들어질 CSV 파일

with open(INPUT, "r", encoding="utf-8") as fin, \
     open(OUTPUT, "w", newline="", encoding="utf-8") as fout:

    writer = None

    for line in fin:
        line = line.strip()
        if not line:
            continue                    # 빈 줄 건너뛰기

        record = json.loads(line)       # JSON → dict

        # 첫 레코드에서 헤더를 동적으로 생성
        if writer is None:
            writer = csv.DictWriter(fout, fieldnames=record.keys())
            writer.writeheader()

        writer.writerow(record)

print(f"✅  Done!  → {OUTPUT}")
