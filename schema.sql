-- Power System Equipment Database Schema

-- Equipment table (base table for all equipment types)
CREATE TABLE IF NOT EXISTS Equipment (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL
);

-- Transformer table
CREATE TABLE IF NOT EXISTS Transformer (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    apparent_power FLOAT NOT NULL,
    voltage_level FLOAT NOT NULL,
    status VARCHAR(20) NOT NULL,
    FOREIGN KEY (id) REFERENCES Equipment(id) ON DELETE CASCADE
);

-- Terminal table
CREATE TABLE IF NOT EXISTS Terminal (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    connected BOOLEAN DEFAULT FALSE
);

-- Substation table
CREATE TABLE IF NOT EXISTS Substation (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    latitude FLOAT,
    longitude FLOAT
);

-- Line table
CREATE TABLE IF NOT EXISTS Line (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    length FLOAT,
    voltage_level FLOAT NOT NULL
);

-- Equipment-Substation relationship table
CREATE TABLE IF NOT EXISTS EquipmentSubstation (
    equipment_id VARCHAR(50),
    substation_id VARCHAR(50),
    PRIMARY KEY (equipment_id, substation_id),
    FOREIGN KEY (equipment_id) REFERENCES Equipment(id) ON DELETE CASCADE,
    FOREIGN KEY (substation_id) REFERENCES Substation(id) ON DELETE CASCADE
);

-- Terminal-Equipment relationship table
CREATE TABLE IF NOT EXISTS TerminalConnection (
    terminal_id VARCHAR(50),
    equipment_id VARCHAR(50),
    PRIMARY KEY (terminal_id, equipment_id),
    FOREIGN KEY (terminal_id) REFERENCES Terminal(id) ON DELETE CASCADE,
    FOREIGN KEY (equipment_id) REFERENCES Equipment(id) ON DELETE CASCADE
);

-- Terminal-Line relationship table
CREATE TABLE IF NOT EXISTS TerminalLine (
    terminal_id VARCHAR(50),
    line_id VARCHAR(50),
    PRIMARY KEY (terminal_id, line_id),
    FOREIGN KEY (terminal_id) REFERENCES Terminal(id) ON DELETE CASCADE,
    FOREIGN KEY (line_id) REFERENCES Line(id) ON DELETE CASCADE
);

-- Sample data for testing

-- Insert Equipment
INSERT INTO Equipment (id, name, description, type) VALUES
('EQ001', 'Main Power Equipment 1', 'High voltage transmission equipment', 'Transformer'),
('EQ002', 'Distribution Equipment 2', 'Medium voltage distribution equipment', 'Transformer'),
('EQ003', 'Auxiliary Equipment 3', 'Low voltage auxiliary equipment', 'Transformer'),
('EQ004', 'Transmission Line Equipment', 'Long-distance transmission equipment', 'Line'),
('EQ005', 'Connection Terminal A', 'Terminal connection point', 'Terminal'),
('EQ006', 'Connection Terminal B', 'Terminal connection point', 'Terminal'),
('EQ007', 'Connection Terminal C', 'Terminal connection point', 'Terminal');

-- Insert Transformers
INSERT INTO Transformer (id, name, apparent_power, voltage_level, status) VALUES
('EQ001', 'Main Power Transformer 1', 100.0, 500.0, 'active'),
('EQ002', 'Distribution Transformer 2', 50.0, 230.0, 'active'),
('EQ003', 'Auxiliary Transformer 3', 25.0, 115.0, 'standby');

-- Insert Terminals
INSERT INTO Terminal (id, name, connected) VALUES
('T001', 'Terminal A', TRUE),
('T002', 'Terminal B', TRUE),
('T003', 'Terminal C', FALSE),
('T004', 'Terminal D', TRUE),
('T005', 'Terminal E', TRUE);

-- Insert Substations
INSERT INTO Substation (id, name, latitude, longitude) VALUES
('S001', 'Main Substation North', 42.3601, -71.0589),
('S002', 'Distribution Substation East', 40.7128, -74.0060),
('S003', 'Auxiliary Substation South', 37.7749, -122.4194);

-- Insert Lines
INSERT INTO Line (id, name, length, voltage_level) VALUES
('L001', 'Transmission Line 1', 150.5, 500.0),
('L002', 'Distribution Line 2', 75.2, 230.0),
('L003', 'Connection Line 3', 25.8, 115.0);

-- Create relationships
-- Equipment in Substations
INSERT INTO EquipmentSubstation (equipment_id, substation_id) VALUES
('EQ001', 'S001'),
('EQ002', 'S002'),
('EQ003', 'S003'),
('EQ004', 'S001');

-- Terminal connections
INSERT INTO TerminalConnection (terminal_id, equipment_id) VALUES
('T001', 'EQ001'),
('T002', 'EQ002'),
('T003', 'EQ003'),
('T004', 'EQ001');

-- Terminal to Line connections
INSERT INTO TerminalLine (terminal_id, line_id) VALUES
('T001', 'L001'),
('T002', 'L002'),
('T004', 'L003');