# -*- coding: utf-8 -*-
"""
Google Drive utilities for file downloading and extraction.
"""

import os
import io
import zipfile
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def get_drive_service():
    """Authenticates with Google Drive API and returns a service object."""
    creds = None
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    TOKEN_FILE, CREDENTIALS_FILE = 'token.json', 'credentials.json'

    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except ValueError:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES,
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
            )
            auth_url, _ = flow.authorization_url(access_type='offline', prompt='consent')
            print(f'\nPlease go to this URL to authorize the application:\n{auth_url}\n')
            code = input('Enter the authorization code here: ').strip()
            flow.fetch_token(code=code)
            creds = flow.credentials

        with open(TOKEN_FILE, 'w') as token_file:
            token_file.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

def download_and_extract_zip(file_id: str, extract_to: str):
    """Downloads and extracts a ZIP file from Google Drive."""
    os.makedirs(extract_to, exist_ok=True)
    print("INFO: Initializing Google Drive service...")
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)

    print("INFO: Downloading file from Google Drive...")
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download Progress: {int(status.progress() * 100)}%")

    buffer.seek(0)
    with zipfile.ZipFile(buffer) as z:
        print("INFO: Extracting files...")
        z.extractall(path=extract_to)
    print(f"SUCCESS: All files extracted to `{extract_to}` directory.")
