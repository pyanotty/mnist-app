FROM python:3.8-buster

WORKDIR /app

RUN pip install uvicorn
RUN pip install fastapi
RUN pip install python-multipart

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]