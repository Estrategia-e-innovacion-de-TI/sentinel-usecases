"""
ZTNA Log Parser

Parser for Cloudflare Zero Trust Network Access (ZTNA) logs.
Handles JSON format logs that may be concatenated without separators.
"""

import re
import json
import pandas as pd
from .base_parser import BaseLogParser


# Pattern to match JSON objects in concatenated format
JSON_PATTERN = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')


class ZTNAParser(BaseLogParser):
    """
    ZTNAParser is a log parser class that extends the BaseLogParser to parse 
    Zero Trust Network Access (ZTNA) logs from Cloudflare.
    
    The logs are in JSON format, often concatenated without separators.
    Each log entry contains network traffic information including actions,
    source/destination IPs, ports, user information, and policy details.
    """
    
    def parse(self) -> pd.DataFrame:
        """
        Reads the ZTNA log file, extracts JSON objects, and returns structured data.

        Returns:
            pd.DataFrame: A DataFrame containing the parsed log data with columns:
                - Action: The action taken (e.g., 'blockByRule', 'allowedByRule')
                - Datetime: Timestamp of the event in ISO format
                - DestinationIP: Destination IP address
                - DestinationPort: Destination port number
                - DeviceID: Unique device identifier
                - DeviceName: Name of the device
                - Email: User email address
                - PolicyID: Policy identifier that triggered the action
                - PolicyName: Name of the policy
                - SessionID: Session identifier
                - SourceIP: Source IP address (external)
                - SourceIPCountryCode: Country code of source IP
                - SourceInternalIP: Internal source IP address
                - SourcePort: Source port number
                - TransportProtocol: Transport protocol (tcp/udp)
                - UserID: User identifier
                - SNI: Server Name Indication (if available)
                - OverrideIP: Override IP (if applicable)
                - OverridePort: Override port (if applicable)
                - RegistrationID: Device registration ID

        Raises:
            Exception: If the file cannot be opened or read, returns empty DataFrame.
        """
        import time
        
        # Retry logic for OneDrive timeouts
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                with open(self.file_path, "r", encoding="utf-8") as file:
                    raw_log = file.read()
                break  # Success, exit retry loop
                
            except OSError as e:
                if "Operation timed out" in str(e) or "timed out" in str(e).lower():
                    if attempt < max_retries - 1:
                        # Retry after delay
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Max retries reached, skip this file
                        print(f"Timeout reading {self.file_path} after {max_retries} attempts, skipping...")
                        return pd.DataFrame()
                else:
                    # Other OS error
                    print(f"Error reading file {self.file_path}: {e}")
                    return pd.DataFrame()
                    
            except Exception as e:
                print(f"Error reading file {self.file_path}: {e}")
                return pd.DataFrame()

        # Extract all JSON objects from the concatenated log
        json_matches = JSON_PATTERN.findall(raw_log)
        
        if not json_matches:
            # File might be empty or not contain JSON
            return pd.DataFrame()
        
        data = []
        for json_str in json_matches:
            try:
                log_entry = json.loads(json_str)
                data.append({
                    'Action': log_entry.get('Action'),
                    'Datetime': log_entry.get('Datetime'),
                    'DestinationIP': log_entry.get('DestinationIP'),
                    'DestinationPort': log_entry.get('DestinationPort'),
                    'DeviceID': log_entry.get('DeviceID'),
                    'DeviceName': log_entry.get('DeviceName'),
                    'Email': log_entry.get('Email'),
                    'PolicyID': log_entry.get('PolicyID'),
                    'PolicyName': log_entry.get('PolicyName'),
                    'SessionID': log_entry.get('SessionID'),
                    'SourceIP': log_entry.get('SourceIP'),
                    'SourceIPCountryCode': log_entry.get('SourceIPCountryCode'),
                    'SourceInternalIP': log_entry.get('SourceInternalIP'),
                    'SourcePort': log_entry.get('SourcePort'),
                    'TransportProtocol': log_entry.get('TransportProtocol'),
                    'UserID': log_entry.get('UserID'),
                    'SNI': log_entry.get('SNI'),
                    'OverrideIP': log_entry.get('OverrideIP'),
                    'OverridePort': log_entry.get('OverridePort'),
                    'RegistrationID': log_entry.get('RegistrationID')
                })
            except json.JSONDecodeError:
                # Skip malformed JSON entries
                continue
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
