# IoT Spam Control - Database Documentation

## Tables

### traffic_data
- id: PRIMARY KEY
- timestamp: DATETIME
- source_ip: VARCHAR(45)
- dest_ip: VARCHAR(45)
- protocol: VARCHAR(20)
- packet_size: INT
- is_spam: BOOLEAN

### spam_patterns
- id: PRIMARY KEY
- pattern_hash: VARCHAR(64)
- pattern_type: VARCHAR(50)
- confidence: FLOAT
- created_at: DATETIME

### detection_logs
- id: PRIMARY KEY
- traffic_id: FOREIGN KEY
- detection_score: FLOAT
- is_true_positive: BOOLEAN
- reviewed_at: DATETIME

## Indexes
- traffic_data(timestamp)
- spam_patterns(pattern_hash)
- detection_logs(traffic_id)
