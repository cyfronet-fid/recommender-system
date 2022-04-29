FROM tiangolo/uwsgi-nginx-flask:python3.10

WORKDIR /app
COPY . /app

RUN pip install pipenv
RUN pipenv lock --dev -r > requirements.txt
RUN pip install -r requirements.txt

ENV UWSGI_INI /app/uwsgi.ini
ENV FLASK_ENV=production
ENV FLASK_APP=/app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV MONGODB_HOST=mongo:27017
ENV REDIS_HOST=redis:6379

EXPOSE 5000