-- Flight Metrics Database Schema
-- MySQL 8.0+

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS ratings;
DROP TABLE IF EXISTS team_draft_results;
DROP TABLE IF EXISTS listen_rankings;
DROP TABLE IF EXISTS flights;

-- Main flights table
CREATE TABLE flights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    origin VARCHAR(3) NOT NULL,
    destination VARCHAR(3) NOT NULL,
    departure_time DATETIME NOT NULL,
    arrival_time DATETIME NOT NULL,
    duration VARCHAR(20) NOT NULL,
    stops INT NOT NULL DEFAULT 0,
    price FLOAT NOT NULL,
    dis_from_origin FLOAT,
    dis_from_dest FLOAT,
    departure_seconds INT,
    arrival_seconds INT,
    duration_min FLOAT,
    date_retrieved DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_data JSON,
    INDEX idx_origin_dest (origin, destination),
    INDEX idx_departure (departure_time),
    INDEX idx_price (price)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- LISTEN ranking evaluations
CREATE TABLE listen_rankings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    prompt TEXT NOT NULL,
    flight_ids JSON NOT NULL,
    user_ranking JSON NOT NULL,
    mode VARCHAR(50) DEFAULT 'listen',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    INDEX idx_user (user_id),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Team Draft evaluation results
CREATE TABLE team_draft_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    prompt TEXT NOT NULL,
    algorithm_a VARCHAR(100) NOT NULL,
    algorithm_b VARCHAR(100) NOT NULL,
    algorithm_a_ranking JSON NOT NULL,
    algorithm_b_ranking JSON NOT NULL,
    interleaved_list JSON NOT NULL,
    user_preferences JSON NOT NULL,
    a_score INT DEFAULT 0,
    b_score INT DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    INDEX idx_user (user_id),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Individual flight ratings
CREATE TABLE ratings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    flight_id INT NOT NULL,
    rating INT NOT NULL,
    prompt_giver BOOLEAN DEFAULT FALSE,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (user_id),
    INDEX idx_flight (flight_id),
    INDEX idx_timestamp (timestamp),
    FOREIGN KEY (flight_id) REFERENCES flights(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
