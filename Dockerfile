FROM python:3.6-alpine
MAINTAINER Christian Dallago "code@dallago.us"

COPY ./requirements.txt /app/requirements.txt

RUN apk add make automake gcc g++ subversion python3-dev libffi-dev freetype-dev libpng-dev

WORKDIR /app
RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["python"]
CMD ["app/main.py"]