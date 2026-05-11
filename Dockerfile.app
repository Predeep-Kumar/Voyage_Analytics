FROM python:3.10-slim

WORKDIR /app

# Prevent Python cache files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Copy requirements first for better layer caching
COPY requirements.app.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -i https://pypi.org/simple -r requirements.app.txt

# Copy only required application files

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


