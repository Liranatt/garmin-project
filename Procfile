web: uvicorn src.api:app --host 0.0.0.0 --port $PORT
bulk-import: python src/bulk_import.py --auto --overlap-days 7
export-reminder: python src/send_export_reminder.py
