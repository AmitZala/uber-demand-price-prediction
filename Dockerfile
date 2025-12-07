# set up the base image
FROM python:3.10-slim

# set the working directory
WORKDIR /app/

# copy the requirements file to workdir
COPY requirements-docker.txt .

# install the requirements
RUN pip install --no-cache-dir -r requirements-docker.txt

# copy the data files needed by the app
COPY ./data/interim/df_without_outliers.csv ./data/interim/df_without_outliers.csv 
COPY ./data/processed/test.csv ./data/processed/test.csv

# copy the models directory
COPY ./models/ ./models/ 

# copy the code files
COPY ./app.py ./app.py

# expose the port on the container
EXPOSE 8501

# run the streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

