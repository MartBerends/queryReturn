# Use the official Python 3.9 slim image
FROM python:3.10-slim


# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
COPY index.html /app/
# Set entry point to main.py
#CMD ["python", "main.py"]

CMD ["gunicorn", "--bind", ":8080", "--threads", "1", "app:app"]
