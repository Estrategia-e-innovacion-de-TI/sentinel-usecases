"""Example: creating a custom log parser by extending BaseLogParser."""

import pandas as pd
from sentinel.ingestion import BaseLogParser


class SimpleCSVLogParser(BaseLogParser):
    """Parser for simple CSV-formatted log files."""

    def parse(self) -> pd.DataFrame:
        """Read a CSV log file into a DataFrame.

        Returns
        -------
        pd.DataFrame
            Parsed log data.
        """
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            print(f"Error reading {self.file_path}: {e}")
            return pd.DataFrame()


def main():
    # Create a sample CSV log file
    sample_data = "timestamp,level,message\n2025-01-01 00:00:00,INFO,Service started\n2025-01-01 00:01:00,ERROR,Connection timeout\n"

    with open("/tmp/sample_log.csv", "w") as f:
        f.write(sample_data)

    # Parse it
    parser = SimpleCSVLogParser("/tmp/sample_log.csv")
    df = parser.parse()
    print(df)
    print(f"\nParsed {len(df)} log entries")


if __name__ == "__main__":
    main()
