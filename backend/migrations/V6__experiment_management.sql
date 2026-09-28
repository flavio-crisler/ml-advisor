ALTER TABLE experiments ADD COLUMN deleted_at DATETIME;

CREATE INDEX idx_experiments_deleted_at ON experiments(deleted_at);
